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

"""
NOTE: rolling update_momenta into p2g: ~205-208FPS -> 224FPS

"""

ti.init(arch=ti.gpu)  # Try to run on GPU

n_grid = 128 * quality
dx = 1 / n_grid  # grid spacing
inv_dx = float(n_grid)
dt = 1e-4 / quality
particles_per_cell_1d = 2
particles_per_cell = particles_per_cell_1d ** 2
p_vol = (dx / particles_per_cell_1d) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E = 5e3  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters


bar_height_grid_cells = int(n_grid / 4)
bar_width_grid_cells = int(n_grid / 2)

n_particles = int(particles_per_cell * bar_height_grid_cells * bar_width_grid_cells)
print("n_particles", n_particles)

bar_height = bar_height_grid_cells * dx
bar_width = bar_width_grid_cells * dx

V_0 = p_vol

particle_field_shape = (2 * bar_width_grid_cells, 2 * bar_height_grid_cells)


x_config = ti.Vector.field(2, dtype=float, shape=particle_field_shape)  # position
x_world = ti.Vector.field(2, dtype=float, shape=particle_field_shape)  # position
v = ti.Vector.field(2, dtype=float, shape=particle_field_shape)  # velocity
F = ti.Matrix.field(
    2, 2, dtype=float, shape=particle_field_shape
)  # deformation gradient
Pk = ti.Matrix.field(2, 2, dtype=float, shape=particle_field_shape)  # pk stresses

W_p2g = ti.Matrix.field(3, 3, dtype=float, shape=particle_field_shape)
W_grad_x_p2g = ti.Matrix.field(3, 3, dtype=float, shape=particle_field_shape)
W_grad_y_p2g = ti.Matrix.field(3, 3, dtype=float, shape=particle_field_shape)


# NOTE: to prevent errors at the boundaries,
# we add a bit of padding around the grid, as well as an offset to compensate.
# The Padding is two extra cells along each edge, so +4 cells horiz and vert.

grid_field_shape = (bar_width_grid_cells + 4, bar_height_grid_cells + 4)
grid_offset = (-2, -2)
grid_v = ti.Vector.field(
    2, dtype=float, shape=grid_field_shape, offset=grid_offset
)  # grid node momentum/velocity
grid_v_next_tmp = ti.Vector.field(
    2, dtype=float, shape=grid_field_shape, offset=grid_offset
)  # grid node momentum/velocity
grid_m = ti.field(
    dtype=float, shape=grid_field_shape, offset=grid_offset
)  # grid node mass
grid_mv = ti.Vector.field(
    2, dtype=float, shape=grid_field_shape, offset=grid_offset
)  # grid forces
grid_f = ti.Vector.field(
    2, dtype=float, shape=grid_field_shape, offset=grid_offset
)  # grid forces

W_stencil = ti.field(dtype=float, shape=(4, 4))
W_stencil_grad_x = ti.field(dtype=float, shape=(4, 4))
W_stencil_grad_y = ti.field(dtype=float, shape=(4, 4))

gravity = ti.Vector.field(2, dtype=float, shape=())
gravity[None] = [0, 0]
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

max_nodal_mass = ti.field(dtype=float, shape=())

alpha = 0.99

# render fields
x_render = ti.Vector.field(
    2, dtype=float, shape=particle_field_shape[0] * particle_field_shape[1]
)  # position


@ti.pyfunc
def particle_index_to_x_config(fg):
    # map particle index to position in configuration space
    return 0.25 * dx + 0.5 * dx * fg.cast(float)


@ti.pyfunc
def grid_index_to_center_pos(i, j):
    # map grid index to cell center position in configuration space
    return dx * (ti.Vector([i, j]).cast(float) + 0.5)


@ti.pyfunc
def particle_index_to_grid_base(f, g):
    return ti.Vector([f // 2, g // 2]) - 1


@ti.pyfunc
def particle_index_to_lower_left_cell_index_in_range(f, g):
    return (f - 1) // 2, (g - 1) // 2


@ti.pyfunc
def grid_index_to_particle_base(f, g):
    return 2 * f - 1, 2 * g - 1


@ti.pyfunc
def W_stencil_index_from_ij_fg(i, j, f, g):
    return (f - (2 * i - 1), g - (2 * j - 1))


@ti.kernel
def init_particle_data():
    # FIXME: this initialization _assumes_ a hardcoded value of 4 particles per cell (2 along any single dimension)
    for f, g in x_config:
        fg = ti.Vector([f, g])
        x_config[f, g] = particle_index_to_x_config(fg)
        bar_center = ti.Vector([0.5, 0.5])
        bar_lower_left = bar_center - 0.5 * ti.Vector([bar_width, bar_height])
        x_world[f, g] = x_config[f, g] + bar_lower_left
        offset_from_center = x_world[f, g] - bar_center
        v[f, g] = 50 * ti.Vector([offset_from_center[1], -offset_from_center[0]])
        F[f, g] = ti.Matrix([[1, 0], [0, 1]])


@ti.pyfunc
def hat_kern_1d(x, x_I):
    # "TLMPM Contacts", eq 2.1
    abs_dist = ti.abs(x - x_I)
    return (1.0 - abs_dist / dx) if abs_dist < dx else 0.0


@ti.pyfunc
def hat_kern_2d(x, x_I):
    # "TLMPM Contacts", eq 2.3
    return hat_kern_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1])


@ti.pyfunc
def hat_kern_derivative_1d(x, x_I):
    # "TLMPM Contacts", eq 2.2
    abs_dist = ti.abs(x - x_I)
    sign = 1.0 if x > x_I else -1.0
    return -sign / dx if abs_dist < dx else 0.0


@ti.pyfunc
def hat_kern_derivative_2d(x, x_I):
    # "TLMPM Contacts", eq 2.3
    return [
        hat_kern_derivative_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1]),
        hat_kern_1d(x[0], x_I[0]) * hat_kern_derivative_1d(x[1], x_I[1]),
    ]


@ti.kernel
def compute_p2g_weights_and_grads():
    for f, g in x_config:
        base = particle_index_to_grid_base(f, g)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            grid_pos = grid_index_to_center_pos(base[0] + i, base[1] + j)
            W_p2g[f, g][i, j] = hat_kern_2d(x_config[f, g], grid_pos)
            W_grad = hat_kern_derivative_2d(x_config[f, g], grid_pos)
            W_grad_x_p2g[f, g][i, j] = W_grad[0]
            W_grad_y_p2g[f, g][i, j] = W_grad[1]


@ti.kernel
def init_cell_stencil_weights_and_grads():
    # we can choose an arbitrary grid node (we chose (1,1) here)
    # and use that (along with initialized x_config values) to find the
    # weights and grads for any cell center to the particle locations in
    # it's neighborhood
    grid_pos = grid_index_to_center_pos(1, 1)
    f_base, g_base = grid_index_to_particle_base(1, 1)
    for f_off, g_off in ti.static(ti.ndrange(4, 4)):
        x_pos = x_config[f_off + f_base, g_off + g_base]
        W_stencil[f_off, g_off] = hat_kern_2d(x_pos, grid_pos)
        W_grad = hat_kern_derivative_2d(x_pos, grid_pos)
        W_stencil_grad_x[f_off, g_off] = W_grad[0]
        W_stencil_grad_y[f_off, g_off] = W_grad[1]


@ti.kernel
def compute_nodal_mass():
    for f, g in x_config:
        base = particle_index_to_grid_base(f, g)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_m[base + offset] += W_p2g[f, g][i, j] * p_mass
    for i, j in grid_m:
        ti.atomic_max(max_nodal_mass[None], grid_m[i, j])


# "TLMPM Contacts", Alg. 1, line 2
init_particle_data()
# "TLMPM Contacts", Alg. 1, line 4
compute_p2g_weights_and_grads()
init_cell_stencil_weights_and_grads()
# "TLMPM Contacts", Alg. 1, line 3
compute_nodal_mass()


@ti.kernel
def p2g():
    # NOTE: in the gather version, we can also do the grid reset within this kernel
    # "TLMPM Contacts", Alg. 1, line 7-12
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]
        grid_f[i, j] = [0, 0]
        f_base, g_base = grid_index_to_particle_base(i, j)
        for f_off, g_off in ti.static(ti.ndrange(4, 4)):

            f = f_base + f_off
            g = g_base + g_off

            weighted_mass = p_mass * W_stencil[f_off, g_off]
            grid_mv[i, j] += weighted_mass * v[f, g]
            force_external = weighted_mass * gravity[None]
            # "TLMPM Contacts", Alg. 1, line 11 ;
            weight_grad = ti.Vector(
                [
                    W_stencil_grad_x[f_off, g_off],
                    W_stencil_grad_y[f_off, g_off],
                ]
            )
            force_internal = -V_0 * Pk[f, g] @ weight_grad
            # "TLMPM Contacts", Alg. 1, line 12
            grid_f[i, j] += force_external + force_internal
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_mv[i, j] / grid_m[i, j]
            # "TLMPM Contacts", Alg. 1, line 14 combined with 17
            grid_v_next_tmp[i, j] = grid_v[i, j] + grid_f[i, j] * dt / grid_m[i, j]


@ti.kernel
def update_particle_velocity():
    # "TLMPM Contacts", Alg. 1, line 18
    for f, g in v:
        v_p = v[f, g] * alpha
        i_base, j_base = particle_index_to_lower_left_cell_index_in_range(f, g)
        for i_off, j_off in ti.static(ti.ndrange(2, 2)):
            i = i_base + i_off
            j = j_base + j_off
            v_next = grid_v_next_tmp[i, j]
            v_this = grid_v[i, j]
            f_stencil, g_stencil = W_stencil_index_from_ij_fg(i, j, f, g)
            weight = W_stencil[f_stencil, g_stencil]
            v_p += alpha * weight * (v_next - v_this) + (1 - alpha) * weight * v_next
        v[f, g] = v_p


@ti.kernel
def update_grid_velocity():
    # "TLMPM Contacts", Alg. 1, line 19
    for i, j in grid_m:
        this_grid_mv = ti.Vector([0.0, 0.0])
        f_base, g_base = grid_index_to_particle_base(i, j)
        for f_off, g_off in ti.static(ti.ndrange(4, 4)):
            f = f_base + f_off
            g = g_base + g_off
            this_grid_mv += p_mass * W_stencil[f_off, g_off] * v[f, g]
        if grid_m[i, j] > 0:
            grid_v[i, j] = this_grid_mv / grid_m[i, j]


@ti.kernel
def g2p():
    for f, g in x_config:
        i_base, j_base = particle_index_to_lower_left_cell_index_in_range(f, g)

        # we don't actually need to loop over a 3x3 neighborhood;
        # for a kernel with radius dx (i.e with radius equal to grid spacing)
        # only _4_ grid nodes (2x2) will ever be in range
        for i_off, j_off in ti.static(ti.ndrange(2, 2)):
            i = i_base + i_off
            j = j_base + j_off

            f_stencil, g_stencil = W_stencil_index_from_ij_fg(i, j, f, g)

            v_ij = grid_v[i, j]
            weight = W_stencil[f_stencil, g_stencil]
            # "TLMPM Contacts", Alg. 1, line 23-24; see MPM after 25 yrs, eq 2.70
            weight_grad = ti.Vector(
                [
                    W_stencil_grad_x[f_stencil, g_stencil],
                    W_stencil_grad_y[f_stencil, g_stencil],
                ]
            )
            F[f, g] += dt * v_ij.outer_product(weight_grad)

            # NOTE: skipping "TLMPM Contacts", Alg. 1, line 25-26;
            #  don't seem to be needed for Neo-Hookean constitutive model

            # "TLMPM Contacts", Alg. 1, line 28
            x_world[f, g] += dt * weight * v_ij

        # "TLMPM Contacts", Alg. 1, line 27-ish;
        # NOTE: computing PK stress, but using TLMPM 2020, eq 24 (Neo-Hookean)
        # NOTE: skipping "TLMPM Contacts", Alg. 1, line 25-26;
        # don't seem to be needed for Neo-Hookean constitutive model
        F_inv_trans = F[f, g].inverse().transpose()
        J = F[f, g].determinant()

        Pk[f, g] = mu_0 * (F[f, g] - F_inv_trans) + lambda_0 * ti.log(J) * F_inv_trans


@ti.kernel
def update_render_buffer():
    for f, g in x_world:
        x_render[f + g * particle_field_shape[0]] = x_world[f, g]


res = (512, 512)
window = ti.ui.Window("Taichi TLMPM", res=res)
canvas = window.get_canvas()
radius = 0.002

frame = 0
duration = 0
fps_counter = FpsCounter()
while window.running and duration < max_duration:
    fps, duration = fps_counter.count_fps(frame)
    frame += 1
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            init_particle_data()
        elif window.event.key in [ti.ui.ESCAPE]:
            break
    for s in range(int(2e-3 // dt)):
        p2g()
        update_particle_velocity()
        update_grid_velocity()
        g2p()
    update_render_buffer()

    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(x_render, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
