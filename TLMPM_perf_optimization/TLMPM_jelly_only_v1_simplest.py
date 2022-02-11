import taichi as ti

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
bar_height_grid_cells = n_grid / 4
bar_width_grid_cells = n_grid / 2

n_particles = int(particles_per_cell * bar_height_grid_cells * bar_width_grid_cells)
print("n_particles", n_particles)

bar_height = bar_height_grid_cells * dx
bar_width = bar_width_grid_cells * dx

V_0 = p_vol
########


x_config = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
x_world = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Pk = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # pk stresses

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

mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))


################################################################################
# init
################################################################################


@ti.kernel
def initialization():
    # FIXME: this initialization _assumes_ a hardcoded value of 4 particles per cell (2 along any single dimension)
    for p in range(n_particles):
        # NOTE: "2.0"s in expression below are to account for 2 points per cell
        # horizontally*vertically = 4 points per cell
        i = int(p % (bar_width_grid_cells * 2.0))
        j = int(p / (bar_width_grid_cells * 2.0))
        # NOTE: "0.25 * dx" is to offset points to interior of cells (to quadrature points)
        x_config[p] = [
            0.25 * dx + 0.5 + i * dx / 2.0 - 0.5 * bar_width,
            0.25 * dx + 0.5 + j * dx / 2.0 - 0.5 * bar_height,
        ]
        x_world[p] = x_config[p]
        center_offset = x_config[p] - ti.Vector([0.5, 0.5])
        v[p] = 50.0 * ti.Vector([center_offset[1], -center_offset[0]])
        F[p] = ti.Matrix([[1, 0], [0, 1]])


@ti.pyfunc
def hat_kern_1d(x, x_I):
    # TLMPM contacts, eq 2.1
    abs_dist = ti.abs(x - x_I)
    return (1.0 - abs_dist / dx) if abs_dist < dx else 0.0


@ti.pyfunc
def hat_kern_2d(x, x_I):
    # TLMPM contacts, eq 2.3
    return hat_kern_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1])


@ti.pyfunc
def hat_kern_derivative_1d(x, x_I):
    # TLMPM contacts, eq 2.2
    abs_dist = ti.abs(x - x_I)
    sign = 1.0 if x > x_I else -1.0
    return -sign / dx if abs_dist < dx else 0.0


@ti.pyfunc
def hat_kern_derivative_2d(x, x_I):
    # TLMPM contacts, eq 2.3
    return [
        hat_kern_derivative_1d(x[0], x_I[0]) * hat_kern_1d(x[1], x_I[1]),
        hat_kern_1d(x[0], x_I[0]) * hat_kern_derivative_1d(x[1], x_I[1]),
    ]


@ti.pyfunc
def print_weight_diagnostics(p, i, j):
    print("p:    ", p)
    print("x_config:    ", x_config[p])
    print("W_p2g:    ", W_p2g[p])
    print("W_grad_x_p2g:    ", W_grad_x_p2g[p])
    print("W_grad_y_p2g:    ", W_grad_y_p2g[p])
    base = ti.floor(x_config[p] * inv_dx - 0.5)
    print("base:    ", base)
    print("F:    ", F[p])
    print("Pk:    ", Pk[p])

    offset = ti.Vector([i, j])
    grid_pos = (base + offset) * dx

    print("grid_pos:    ", grid_pos)
    print("x_config[p] - grid_pos:    ", x_config[p] - grid_pos)
    print("hat_kern_2d(x_config[p], grid_pos):    ", hat_kern_2d(x_config[p], grid_pos))
    print(
        "hat_kern_derivative_2d(x_config[p], grid_pos):    ",
        hat_kern_derivative_2d(x_config[p], grid_pos),
    )
    weight_grad = ti.Vector([W_grad_x_p2g[p][i, j], W_grad_y_p2g[p][i, j]])
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
            force_external = weighted_mass * gravity[None]
            # TLMPM contacts, Alg. 1, line 11 ;
            weight_grad = ti.Vector([W_grad_x_p2g[p][i, j], W_grad_y_p2g[p][i, j]])
            force_internal = -V_0 * Pk[p] @ weight_grad
            # force_internal = ti.Vector([0.0, 0.0])
            grid_f[base + offset] += force_external + force_internal


@ti.kernel
def update_momenta():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_mv[i, j] / grid_m[i, j]
    # TLMPM contacts, Alg. 1, line 14, 17
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v_next_tmp[i, j] = grid_v[i, j] + grid_f[i, j] * dt / grid_m[i, j]


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

        v_I = ti.Vector([0.0, 0.0])
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            weight = W_p2g[p][i, j]
            v_I = grid_v[base + offset]

            # TLMPM contacts, Alg. 1, line 23-24; see MPM after 25 yrs, eq 2.70
            weight_grad = ti.Vector([W_grad_x_p2g[p][i, j], W_grad_y_p2g[p][i, j]])
            F[p] += dt * v_I.outer_product(weight_grad)

            # NOTE: skipping TLMPM contacts, Alg. 1, line 25-26;
            #  don't seem to be needed for Neo-Hookean constitutive model

            # TLMPM contacts, Alg. 1, line 28
            x_world[p] += dt * weight * v_I

        # TLMPM contacts, Alg. 1, line 27-ish;
        # NOTE: computing PK stress, but using TLMPM 2020, eq 24 (Neo-Hookean)
        # NOTE: skipping TLMPM contacts, Alg. 1, line 25-26;
        # don't seem to be needed for Neo-Hookean constitutive model
        F_inv_trans = F[p].inverse().transpose()
        J = F[p].determinant()

        Pk[p] = mu_0 * (F[p] - F_inv_trans) + lambda_0 * ti.log(J) * F_inv_trans


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

    canvas.set_background_color((0.067, 0.184, 0.255))
    canvas.circles(x_world, radius=radius, color=(0.93, 0.33, 0.23))
    window.show()
