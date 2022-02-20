import taichi as ti

import argparse, sys, os

sys.path.append(os.path.abspath(__file__ + "/../.."))

from utils.fps_counter import FpsCounter

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--quality", help="simulation quality", type=int)
parser.add_argument("-t", "--time", help="simulation duration (seconds)", type=int)
args = parser.parse_args()
max_duration = args.time if args.time else 60 * 5
quality = (
    args.quality if args.quality else 3
)  # Use a larger value for higher-res simulations

ti.init(arch=ti.gpu)  # Try to run on GPU

n_grid = 128 * quality
dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2  # implies four particles per grid cell
p_rho = 1
p_mass = p_vol * p_rho
E = 5e3  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters


bar_height_grid_cells = n_grid / 4
bar_width_grid_cells = n_grid / 2

n_particles = int(4 * bar_height_grid_cells * bar_width_grid_cells)
print("n_particles", n_particles)

bar_height = bar_height_grid_cells * dx
bar_width = bar_width_grid_cells * dx


x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient

Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

grid_v = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

gravity = ti.Vector.field(2, dtype=float, shape=())
gravity[None] = [0, 0]
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

group_size = n_particles
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))


max_nodal_mass = ti.field(dtype=float, shape=())


@ti.kernel
def reset():
    # NOTE: this initialization _assumes_ a hardcoded value of 4 particles per cell (2 along any single dimension)
    # the "2.0"s in expression below are to account for 2 points per cell
    # horizontally*vertically = 4 points per cell
    for p in range(n_particles):
        i = int(p % (bar_width_grid_cells * 2.0))
        j = int(p / (bar_width_grid_cells * 2.0))
        # NOTE: "0.25 * dx" is to offset points to interior of cells (to quadrature points)
        x[p] = [
            0.25 * dx + 0.5 + i * dx / 2.0 - 0.5 * bar_width,
            0.25 * dx + 0.5 + j * dx / 2.0 - 0.5 * bar_height,
        ]
        center_offset = x[p] - ti.Vector([0.5, 0.5])
        v[p] = 50.0 * ti.Vector([center_offset[1], -center_offset[0]])
        F[p] = ti.Matrix([[1, 0], [0, 1]])
        Jp[p] = 1
        C[p] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def substep():
    # clear grid
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

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
            # grid_v[base + offset] += weight * p_mass * v[p]
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        ti.atomic_max(max_nodal_mass[None], grid_m[i, j])

    # update grid
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += (
                dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            )
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

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


res = (512, 512)
window = ti.ui.Window("Taichi MLS-MPM-128", res=res)
canvas = window.get_canvas()
radius = 0.002

reset()

frame = 0
duration = 0
fps_counter = FpsCounter()
while window.running and duration < max_duration:
    fps, duration = fps_counter.count_fps(frame)
    frame += 1
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            reset()
        elif window.event.key in [ti.ui.ESCAPE]:
            break

    for s in range(int(2e-3 // dt)):
        substep()
    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(x, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
