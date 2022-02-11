import taichi as ti

#
"""
NOTE: in this variant, we use a grid of particles in configutation space.
Since we are assuming 4 fixed particles per cell (2 by 2), we can use a grid of these particles to store them. this will have some memorey overhead in the case of non-rectangular objects, but it will simplify access to the particles corresponding to cells and vice versa.

We will treat the particles as pixels in an image. 
The lower-left-most cell will have:
ij (0,0) and configuration space position = dx * (0.5, 0.5).

That cell will contain particles:
ij = (0,0) and configuration space position = dx * (0.25, .025)
ij = (1,0) and configuration space position = dx * (0.75, .025)
ij = (0,1) and configuration space position = dx * (0.25, .075)
ij = (1,1) and configuration space position = dx * (0.75, .075)

Particles in the left of a cell will have odd i; right even i
Particles in the bottom of a cell will have odd j; top even j

In general, a cell (i,j) will contain particles:
(2*i,2*j); (2*i + 1, 2*j); (2*i, 2*j+1); (2*i+1, 2*j+1)

in general, a particle (f,g) will have position
= 0.25 * dx + .5 * dx * (f,g)

To find the cell contaiing a particle, simple integer division will do.

NOTE: In this variant we also dramatically reduce the number of grid cells we need to allocate compared to the very dumb naive version, since this approach simplifies using a tight bounding grid. Doing this gives ~2x speedup with ~74k particles on quality level 3 (~30 -> ~60 fps).
"""

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 3  # Use a larger value for higher-res simulations
# n_particles = 3000 * quality ** 2
n_grid = 128 * quality
dx = 1 / n_grid  # grid spacing
inv_dx = float(n_grid)
dt = 1e-4 / quality
particles_per_cell_along_one_dimension = 2
particles_per_cell = particles_per_cell_along_one_dimension ** 2
p_vol = (dx / particles_per_cell_along_one_dimension) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E = 5e3  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

########
bar_height_grid_cells = int(n_grid / 4)
bar_width_grid_cells = int(n_grid / 2)

n_particles = int(particles_per_cell * bar_height_grid_cells * bar_width_grid_cells)
print("n_particles", n_particles)

bar_height = bar_height_grid_cells * dx
bar_width = bar_width_grid_cells * dx

V_0 = p_vol
########
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
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))


################################################################################
# init
################################################################################


@ti.pyfunc
def particle_index_to_x_config(ij):
    return 0.25 * dx + 0.5 * dx * ij.cast(float)


@ti.kernel
def initialization():
    # FIXME: this initialization _assumes_ a hardcoded value of 4 particles per cell (2 along any single dimension)
    for i, j in x_config:
        ij = ti.Vector([i, j])
        x_config[i, j] = particle_index_to_x_config(ij)
        bar_center = ti.Vector([0.5, 0.5])
        bar_lower_left = bar_center - 0.5 * ti.Vector([bar_width, bar_height])
        x_world[i, j] = x_config[i, j] + bar_lower_left
        offset_from_center = x_world[i, j] - bar_center
        v[i, j] = 50 * ti.Vector([offset_from_center[1], -offset_from_center[0]])
        F[i, j] = ti.Matrix([[1, 0], [0, 1]])


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


@ti.pyfunc
def print_weight_diagnostics(f, g, i, j):
    print("f,g:    ", f, g)
    print("x_config:    ", x_config[f, g])
    print("W_p2g:    ", W_p2g[f, g])
    print("W_grad_x_p2g:    ", W_grad_x_p2g[f, g])
    print("W_grad_y_p2g:    ", W_grad_y_p2g[f, g])
    base = ti.floor(x_config[f, g] * inv_dx - 0.5)
    print("base:    ", base)
    print("F:    ", F[f, g])
    print("Pk:    ", Pk[f, g])

    offset = ti.Vector([i, j])
    grid_pos = (base + offset) * dx

    print("grid_pos:    ", grid_pos)
    print("x_config[f,g] - grid_pos:    ", x_config[f, g] - grid_pos)
    print(
        "hat_kern_2d(x_config[f,g], grid_pos):    ",
        hat_kern_2d(x_config[f, g], grid_pos),
    )
    print(
        "hat_kern_derivative_2d(x_config[f,g], grid_pos):    ",
        hat_kern_derivative_2d(x_config[f, g], grid_pos),
    )
    weight_grad = ti.Vector([W_grad_x_p2g[f, g][i, j], W_grad_y_p2g[f, g][i, j]])
    print("weight_grad:    ", weight_grad)
    i0, j0 = base + offset
    print("i0, j0:    ", i0, j0)
    v_I = grid_v[i0, j0]
    print("grid_v[i0, j0]:    ", grid_v[i0, j0])
    print("grid_v_next_tmp[i0, j0]:    ", grid_v_next_tmp[i0, j0])
    print("grid_mv[i0, j0]:    ", grid_mv[i0, j0])
    print("grid_m[i0, j0]:    ", grid_m[i0, j0])
    print("grid_f[i0, j0]:    ", grid_f[i0, j0])
    print(
        "dt * v_I.outer_product(weight_grad):    ", dt * v_I.outer_product(weight_grad)
    )


@ti.kernel
def compute_p2g_weights_and_grads():
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_pos = (base + offset).cast(float) * dx
            W_p2g[f, g][i, j] = hat_kern_2d(x_config[f, g], grid_pos)
            W_grad = hat_kern_derivative_2d(x_config[f, g], grid_pos)
            W_grad_x_p2g[f, g][i, j] = W_grad[0]
            W_grad_y_p2g[f, g][i, j] = W_grad[1]


@ti.kernel
def compute_nodal_mass():
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_m[base + offset] += W_p2g[f, g][i, j] * p_mass
    for i, j in grid_m:
        ti.atomic_max(max_nodal_mass[None], grid_m[i, j])


@ti.kernel
def init_grid_v():
    # "TLMPM Contacts", Alg. 1, line 8-12
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_v[base + offset] += W_p2g[f, g][i, j] * v[f, g]


initialization()
compute_p2g_weights_and_grads()
compute_nodal_mass()
init_grid_v()

print(max_nodal_mass[None])

################################################################################
# algorithm steps ("TLMPM Contacts", Alg. 1)
################################################################################


@ti.kernel
def reset_grid():
    # "TLMPM Contacts", Alg. 1, line 7
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]
        grid_f[i, j] = [0, 0]


@ti.kernel
def p2g():
    # "TLMPM Contacts", Alg. 1, line 8-12
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            weighted_mass = p_mass * W_p2g[f, g][i, j]
            grid_mv[base + offset] += weighted_mass * v[f, g]
            force_external = weighted_mass * gravity[None]
            # "TLMPM Contacts", Alg. 1, line 11 ;
            weight_grad = ti.Vector(
                [W_grad_x_p2g[f, g][i, j], W_grad_y_p2g[f, g][i, j]]
            )
            force_internal = -V_0 * Pk[f, g] @ weight_grad
            # force_internal = ti.Vector([0.0, 0.0])
            grid_f[base + offset] += force_external + force_internal


@ti.kernel
def update_momenta():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_mv[i, j] / grid_m[i, j]
    # "TLMPM Contacts", Alg. 1, line 14, 17
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v_next_tmp[i, j] = grid_v[i, j] + grid_f[i, j] * dt / grid_m[i, j]


@ti.kernel
def update_particle_and_grid_velocity():
    # "TLMPM Contacts", Alg. 1, line 18
    for f, g in v:
        v_p = v[f, g] * alpha
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            v_next = grid_v_next_tmp[base + offset]
            v_this = grid_v[base + offset]
            weight = W_p2g[f, g][i, j]
            v_p += alpha * weight * (v_next - v_this) + (1 - alpha) * weight * v_next
        v[f, g] = v_p

    # NOTE: need to reset grid_mv again before Alg.1 line 19
    for i, j in grid_m:
        grid_mv[i, j] = [0, 0]

    # "TLMPM Contacts", Alg. 1, line 19
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_mv[base + offset] += p_mass * W_p2g[f, g][i, j] * v[f, g]

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_mv[i, j] / grid_m[i, j]


@ti.kernel
def g2p():
    for f, g in x_config:
        base = (x_config[f, g] * inv_dx - 0.5).cast(int)

        v_I = ti.Vector([0.0, 0.0])
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            weight = W_p2g[f, g][i, j]
            v_I = grid_v[base + offset]

            # "TLMPM Contacts", Alg. 1, line 23-24; see MPM after 25 yrs, eq 2.70
            weight_grad = ti.Vector(
                [W_grad_x_p2g[f, g][i, j], W_grad_y_p2g[f, g][i, j]]
            )
            F[f, g] += dt * v_I.outer_product(weight_grad)

            # NOTE: skipping "TLMPM Contacts", Alg. 1, line 25-26;
            #  don't seem to be needed for Neo-Hookean constitutive model

            # "TLMPM Contacts", Alg. 1, line 28
            x_world[f, g] += dt * weight * v_I

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
while window.running and frame < 60000:
    frame += 1
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            initialization()
        elif window.event.key in [ti.ui.ESCAPE]:
            break
    for s in range(int(2e-3 // dt)):
        reset_grid()
        p2g()
        update_momenta()
        update_particle_and_grid_velocity()
        g2p()
    update_render_buffer()

    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(x_render, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
