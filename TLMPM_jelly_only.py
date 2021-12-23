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

x_config = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
x_render = ti.Vector.field(2, dtype=float, shape=n_particles)  # position

C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Pk = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # pk stresses
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

W_p2g = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
W_grad_x_p2g = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
W_grad_y_p2g = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)

grid_v = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_v_next_tmp = ti.Vector.field(
    2, dtype=float, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
grid_mv = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid forces
grid_f = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid forces

gravity = ti.Vector.field(2, dtype=float, shape=())
gravity[None] = [0, 0]
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

max_nodal_mass = ti.field(dtype=float, shape=())

alpha = 0.99

# group_size = n_particles // 3
group_size = n_particles
# water = ti.Vector.field(2, dtype=float, shape=group_size)  # position
jelly = ti.Vector.field(2, dtype=float, shape=group_size)  # position
# snow = ti.Vector.field(2, dtype=float, shape=group_size)  # position
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))


v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity

################################################################################
# init
################################################################################


@ti.kernel
def initialization():
    for p in range(n_particles):
        x_config[p] = [
            0.5 + (ti.random() - 0.5) * 0.1,
            0.5 + (ti.random() - 0.5) * 0.3,
            # ti.random() * 0.2 + 0.5 + 0.32 * (i // group_size),
        ]
        x_render[p] = x_config[p]
        # material[p] = i // group_size  # 0: fluid, 1: jelly, 2: snow
        center_offset = x_config[p] - ti.Vector([0.5, 0.5])
        material[p] = 1  # 0: fluid, 1: jelly, 2: snow
        # v[p] = [0, 0]
        v[p] = 0.50 * ti.Vector([center_offset[1], -center_offset[0]])
        F[p] = ti.Matrix([[1, 0], [0, 1]])
        Jp[p] = 1
        C[p] = ti.Matrix.zero(float, 2, 2)


@ti.func
def hat_kern_1d(x, x_I):
    # TLMPM contacts, eq 2.1
    abs_dist = ti.abs(x - x_I)
    return (1 - abs_dist / dx) if abs_dist < dx else 0


@ti.func
def hat_kern_2d(x, x_I):
    # TLMPM contacts, eq 2.3
    return hat_kern_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1])


@ti.func
def hat_kern_derivative_1d(x, x_I):
    # TLMPM contacts, eq 2.2
    abs_dist = ti.abs(x - x_I)
    sign = 1 if x > x_I else 0
    return -sign / dx if abs_dist < dx else 0


@ti.func
def hat_kern_derivative_2d(x, x_I):
    # TLMPM contacts, eq 2.3
    return [
        hat_kern_derivative_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1]),
        hat_kern_1d(x[0], x_I[0]) * hat_kern_derivative_1d(x[1], x_I[1]),
    ]


@ti.kernel
def compute_p2g_weights_and_grads():
    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_pos = (base + offset).cast(float) * dx
            W_p2g[p][i, j] = hat_kern_2d(x_config[p], grid_pos)
            W_grad = hat_kern_derivative_2d(x_config[p], grid_pos)
            W_grad_x_p2g[p][i, j] = W_grad[0]
            W_grad_y_p2g[p][i, j] = W_grad[1]


@ti.kernel
def compute_nodal_mass():
    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_m[base + offset] += W_p2g[p][i, j] * p_mass
    for i, j in grid_m:
        ti.atomic_max(max_nodal_mass[None], grid_m[i, j])


@ti.kernel
def init_grid_v():
    # TLMPM contacts, Alg. 1, line 8-12
    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_v[base + offset] += W_p2g[p][i, j] * v[p]


initialization()
compute_p2g_weights_and_grads()
compute_nodal_mass()
init_grid_v()

print(max_nodal_mass[None])
print(v[0])

################################################################################
# algorithm steps (TLMPM contacts, Alg. 1)
################################################################################


@ti.kernel
def reset_grid():
    # TLMPM contacts, Alg. 1, line 7
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]
        grid_f[i, j] = [0, 0]


@ti.kernel
def p2g():
    # TLMPM contacts, Alg. 1, line 8-12
    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            weighted_mass = p_mass * W_p2g[p][i, j]
            grid_mv[base + offset] += weighted_mass * v[p]
            # force_external = weighted_mass * gravity[None]
            # force_internal = ti.Vector([0.0, 0.0])
            # grid_f[base + offset] += force_external + force_internal


@ti.kernel
def update_momenta():
    # TLMPM contacts, Alg. 1, line 14, 17
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v_next_tmp[i, j] = (grid_mv[i, j] + grid_f[i, j] * dt) / grid_m[i, j]


@ti.kernel
def update_particle_and_grid_velocity():
    # TLMPM contacts, Alg. 1, line 18
    for p in v:
        v_p = v[p] * alpha
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            v_next = grid_v_next_tmp[base + offset]
            v_this = grid_v[base + offset]
            weight = W_p2g[p][i, j]
            v_p += alpha * weight * (v_next - v_this) + (1 - alpha) * weight * v_next
        v[p] = v_p

    # NOTE: need to reset grid_mv again before Alg.1 line 19
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]

    # TLMPM contacts, Alg. 1, line 19
    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_mv[base + offset] += p_mass * W_p2g[p][i, j] * v[p]

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_mv[i, j] / grid_m[i, j]


@ti.kernel
def g2p():

    for p in x_config:
        base = (x_config[p] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            weight = W_p2g[p][i, j]
            # TLMPM contacts, Alg. 1, line 28
            x_render[p] += dt * weight * grid_v[base + offset]


#     # Particle state update and scatter to grid (P2G)
#     for p in x:
#         base = (x[p] * inv_dx - 0.5).cast(int)
#         fx = x[p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
#         F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[
#             p
#         ]  # deformation gradient update

#         h = 1
#         mu, la = mu_0 * h, lambda_0 * h
#         U, sig, V = ti.svd(F[p])
#         J = 1.0
#         for d in ti.static(range(2)):
#             new_sig = sig[d, d]
#             Jp[p] *= sig[d, d] / new_sig
#             sig[d, d] = new_sig
#             J *= new_sig
#         stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[
#             p
#         ].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
#         stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
#         affine = stress + p_mass * C[p]
#         for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
#             offset = ti.Vector([i, j])
#             dpos = (offset.cast(float) - fx) * dx
#             weight = w[i][0] * w[j][1]
#             grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
#             grid_m[base + offset] += weight * p_mass

#     # update grid
#     for i, j in grid_m:
#         if grid_m[i, j] > 0:  # No need for epsilon here
#             grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
#             grid_v[i, j] += dt * gravity[None] * 30  # gravity
#             dist = attractor_pos[None] - dx * ti.Vector([i, j])
#             grid_v[i, j] += (
#                 dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
#             )
#             if i < 3 and grid_v[i, j][0] < 0:
#                 grid_v[i, j][0] = 0  # Boundary conditions
#             if i > n_grid - 3 and grid_v[i, j][0] > 0:
#                 grid_v[i, j][0] = 0
#             if j < 3 and grid_v[i, j][1] < 0:
#                 grid_v[i, j][1] = 0
#             if j > n_grid - 3 and grid_v[i, j][1] > 0:
#                 grid_v[i, j][1] = 0

#     # grid to particle (G2P)
#     for p in x:
#         base = (x[p] * inv_dx - 0.5).cast(int)
#         fx = x[p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
#         new_v = ti.Vector.zero(float, 2)
#         new_C = ti.Matrix.zero(float, 2, 2)
#         for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
#             dpos = ti.Vector([i, j]).cast(float) - fx
#             g_v = grid_v[base + ti.Vector([i, j])]
#             weight = w[i][0] * w[j][1]
#             new_v += weight * g_v
#             new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
#         v[p], C[p] = new_v, new_C
#         x[p] += dt * v[p]  # advection


@ti.kernel
def render():
    for i in range(group_size):
        jelly[i] = x_render[i]
        # water[i] = x[i]
        # jelly[i] = x[i + group_size]
        # snow[i] = x[i + 2 * group_size]


# print(
#     "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to initialization."
# )

res = (512, 512)
window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
canvas = window.get_canvas()
radius = 0.003

# initialization()
# gravity[None] = [0, -1]


while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            initialization()
        elif window.event.key in [ti.ui.ESCAPE]:
            break
    #     # if window.event is not None:
    #     #     gravity[None] = [0, 0]  # if had any event
    #     # if window.is_pressed(ti.ui.LEFT, "a"):
    #     #     gravity[None][0] = -1
    #     # if window.is_pressed(ti.ui.RIGHT, "d"):
    #     #     gravity[None][0] = 1
    #     # if window.is_pressed(ti.ui.UP, "w"):
    #     #     gravity[None][1] = 1
    #     # if window.is_pressed(ti.ui.DOWN, "s"):
    #     #     gravity[None][1] = -1
    #     mouse = window.get_cursor_pos()
    #     mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
    #     canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.05)
    #     attractor_pos[None] = [mouse[0], mouse[1]]
    #     attractor_strength[None] = 0
    #     if window.is_pressed(ti.ui.LMB):
    #         attractor_strength[None] = 1
    #     if window.is_pressed(ti.ui.RMB):
    #         attractor_strength[None] = -1

    # print(x_config[0])
    # print(x_render[0])
    # print(v[0])

    for s in range(int(2e-3 // dt)):
        reset_grid()
        p2g()
        update_momenta()
        # # try:
        update_particle_and_grid_velocity()
        # except:
        #     pass
        g2p()

    render()
    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
