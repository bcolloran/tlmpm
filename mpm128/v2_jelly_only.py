"""
Changes in this version:
* remove the materials other than jelly, move the jelly square
* remove `render()` kernel and the associated fields -- not needed with only one material
* set gravity constant and remove the ability to change gravity
* remove "attractor" and mouse interactions

"""
import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
n_particles = 3000 * quality ** 2
n_grid = 128 * quality
dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E = 5e3  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

# particle fields
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

# grid fields
grid_v = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass


gravity = ti.Vector.field(2, dtype=float, shape=())
gravity[None] = [0, -1]


@ti.kernel
def reset_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0


@ti.kernel
def p2g():
    # Particle state update and scatter to grid (P2G)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[
            p
        ]  # deformation gradient update
        h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[
            p
        ].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass


@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0


@ti.kernel
def g2p():
    # grid to particle (G2P)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.10 * (i // n_particles),
            ti.random() * 0.2 + 0.5 + 0.32 * (i // n_particles),
        ]
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


res = (512, 512)
window = ti.ui.Window("Taichi MLS-MPM-128", res=res)
canvas = window.get_canvas()
radius = 0.003

reset()
gravity[None] = [0, -1]

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            reset()
        elif window.event.key in [ti.ui.ESCAPE]:
            break
    for s in range(int(2e-3 // dt)):
        reset_grid()
        p2g()
        update_grid()
        g2p()
    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(x, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
