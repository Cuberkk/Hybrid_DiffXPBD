import warp as wp

# ---------------------------------------------------------
# custom fixed-length vector types
# ---------------------------------------------------------
vec12 = wp.types.vector(length=12, dtype=wp.float32)
vec45f = wp.types.vector(length=45, dtype=wp.float32)
vec2f = wp.vec2
mat22 = wp.mat22
mat33 = wp.mat33
vec3f = wp.vec3
vec4i = wp.vec4i


# =========================================================
# helper funcs
# =========================================================

@wp.func
def flatten_vec3_4(v1: vec3f, v2: vec3f, v3: vec3f, v4: vec3f):
    out = vec12()
    out[0]  = v1[0]
    out[1]  = v1[1]
    out[2]  = v1[2]

    out[3]  = v2[0]
    out[4]  = v2[1]
    out[5]  = v2[2]

    out[6]  = v3[0]
    out[7]  = v3[1]
    out[8]  = v3[2]

    out[9]  = v4[0]
    out[10] = v4[1]
    out[11] = v4[2]
    return out


@wp.func
def local_B_inv_dot(vec_in: vec12, b0: float, b1: float, b2: float, b3: float):
    out = vec12()

    out[0]  = b0 * vec_in[0]
    out[1]  = b0 * vec_in[1]
    out[2]  = b0 * vec_in[2]

    out[3]  = b1 * vec_in[3]
    out[4]  = b1 * vec_in[4]
    out[5]  = b1 * vec_in[5]

    out[6]  = b2 * vec_in[6]
    out[7]  = b2 * vec_in[7]
    out[8]  = b2 * vec_in[8]

    out[9]  = b3 * vec_in[9]
    out[10] = b3 * vec_in[10]
    out[11] = b3 * vec_in[11]

    return out


@wp.func
def local_quad_form(a: vec12, b: vec12, b0: float, b1: float, b2: float, b3: float):
    val = float(0.0)

    val += b0 * (a[0]  * b[0]  + a[1]  * b[1]  + a[2]  * b[2])
    val += b1 * (a[3]  * b[3]  + a[4]  * b[4]  + a[5]  * b[5])
    val += b2 * (a[6]  * b[6]  + a[7]  * b[7]  + a[8]  * b[8])
    val += b3 * (a[9]  * b[9]  + a[10] * b[10] + a[11] * b[11])

    return val


# =========================================================
# gradient funcs
# =========================================================

@wp.func
def gradient_neo_hookean_H(F: mat33, detF: float, Dm_inv: mat33):
    # P = det(F) * F^{-T}
    P = detF * wp.transpose(wp.inverse(F))
    H = P * wp.transpose(Dm_inv)

    partial_x1 = vec3f(H[0, 0], H[1, 0], H[2, 0])
    partial_x2 = vec3f(H[0, 1], H[1, 1], H[2, 1])
    partial_x3 = vec3f(H[0, 2], H[1, 2], H[2, 2])
    partial_x4 = -(partial_x1 + partial_x2 + partial_x3)

    return flatten_vec3_4(partial_x1, partial_x2, partial_x3, partial_x4)


@wp.func
def gradient_neo_hookean_D(F: mat33, Dm_inv: mat33):
    frob = wp.sqrt(wp.trace(wp.transpose(F) * F) + 1.0e-12)

    P = F / frob
    H = P * wp.transpose(Dm_inv)

    partial_x1 = vec3f(H[0, 0], H[1, 0], H[2, 0])
    partial_x2 = vec3f(H[0, 1], H[1, 1], H[2, 1])
    partial_x3 = vec3f(H[0, 2], H[1, 2], H[2, 2])
    partial_x4 = -(partial_x1 + partial_x2 + partial_x3)

    return flatten_vec3_4(partial_x1, partial_x2, partial_x3, partial_x4)

# =========================================================
# forward local solve: one thread per tetra
# =========================================================

@wp.kernel
def solve_constraints_local_kernel(
    topo: wp.array(dtype=vec4i),              # (nt,)
    x_old: wp.array(dtype=vec3f),             # (nv,)
    Dm_inv_all: wp.array(dtype=mat33),        # (nt,)
    k_inv: wp.array(dtype=wp.float32),        # (nv,)
    lambda_old: wp.array(dtype=vec2f),        # (nt,)
    gamma_: wp.array(dtype=wp.float32),       # shape (1,)
    compliance: wp.array(dtype=mat22),        # (nt,)
    delta_x_local: wp.array(dtype=vec12),     # (nt,)
    delta_lambda_local: wp.array(dtype=vec2f),# (nt,)
    eps: float,
):
    i = wp.tid()

    tet = topo[i]
    id0 = tet[0]
    id1 = tet[1]
    id2 = tet[2]
    id3 = tet[3]

    v1 = x_old[id0]
    v2 = x_old[id1]
    v3 = x_old[id2]
    v4 = x_old[id3]

    Ds = wp.matrix_from_cols(v1 - v4, v2 - v4, v3 - v4)
    Dm_inv = Dm_inv_all[i]
    F = Ds * Dm_inv

    J = wp.determinant(F)
    if J < eps:
        J = eps

    H_compliance = compliance[i][0, 0]
    D_compliance = compliance[i][1, 1]

    constraint_H = J - gamma_[0]
    constraint_D = wp.sqrt(wp.trace(wp.transpose(F) * F)) - wp.sqrt(3.0)

    dCH = gradient_neo_hookean_H(F, J, Dm_inv)
    dCD = gradient_neo_hookean_D(F, Dm_inv)

    b0 = k_inv[id0]
    b1 = k_inv[id1]
    b2 = k_inv[id2]
    b3 = k_inv[id3]

    A00 = local_quad_form(dCH, dCH, b0, b1, b2, b3) + H_compliance
    A01 = local_quad_form(dCH, dCD, b0, b1, b2, b3)
    A11 = local_quad_form(dCD, dCD, b0, b1, b2, b3) + D_compliance

    lam = lambda_old[i]

    rhs0 = -(constraint_H + H_compliance * lam[0])
    rhs1 = -(constraint_D + D_compliance * lam[1])

    detA = A00 * A11 - A01 * A01
    inv_det = 1.0 / detA

    delta_lambda = vec2f(
        ( A11 * rhs0 - A01 * rhs1) * inv_det,
        (-A01 * rhs0 + A00 * rhs1) * inv_det,
    )

    grad_combined = delta_lambda[0] * dCH + delta_lambda[1] * dCD
    dx = local_B_inv_dot(grad_combined, b0, b1, b2, b3)

    delta_x_local[i] = dx
    delta_lambda_local[i] = delta_lambda

@wp.kernel
def solve_local_elastic_force(
    topo: wp.array(dtype=vec4i),              # (nt,)
    x_old: wp.array(dtype=vec3f),             # (nv,)
    Dm_inv_all: wp.array(dtype=mat33),        # (nt,)
    gamma_: wp.array(dtype=wp.float32),       # shape (1,)
    compliance: wp.array(dtype=mat22),        # (nt,)
    compliance_modulation:float,
    local_elastic_force: wp.array(dtype=vec12),     # (nt,)
    eps: float,
    dt: float,
):
    i = wp.tid()

    tet = topo[i]
    id0 = tet[0]
    id1 = tet[1]
    id2 = tet[2]
    id3 = tet[3]

    v1 = x_old[id0]
    v2 = x_old[id1]
    v3 = x_old[id2]
    v4 = x_old[id3]

    Ds = wp.matrix_from_cols(v1 - v4, v2 - v4, v3 - v4)
    Dm_inv = Dm_inv_all[i]
    F = Ds * Dm_inv

    J = wp.determinant(F)
    if J < eps:
        J = eps

    H_compliance = compliance[i][0, 0]
    D_compliance = compliance[i][1, 1]

    H_alpha_inv = 1.0 / (H_compliance * dt * dt / compliance_modulation)
    D_alpha_inv = 1.0 / (D_compliance * dt * dt / compliance_modulation)

    constraint_H = J - gamma_[0]
    constraint_D = wp.sqrt(wp.trace(wp.transpose(F) * F)) - wp.sqrt(3.0)

    dCH = gradient_neo_hookean_H(F, J, Dm_inv)
    dCD = gradient_neo_hookean_D(F, Dm_inv)
    local_elastic_force[i] = -((dCH * H_alpha_inv * constraint_H) + (dCD * D_alpha_inv * constraint_D))

# =========================================================
# host-side wrapper
# =========================================================

class EnergyConstraintSolver3D_Warp:
    def __init__(self, device=None, eps=1.0e-9, dt=1.0e-2):
        self.device = device if device is not None else wp.get_preferred_device()
        self.youngs_modulus = 9.5e3 # unit KPa
        self.poisson_ratio = 0.45
        self.miu = self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
        self.lam = self.youngs_modulus * self.poisson_ratio / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        self.eps = float(eps)
        self.dt = float(dt)

    def alloc_local_buffers(self, nt: int):
        delta_x_local = wp.zeros(shape=(nt,), dtype=vec12, device=self.device)
        delta_lambda_local = wp.zeros(shape=(nt,), dtype=vec2f, device=self.device)
        return delta_x_local, delta_lambda_local

    def alloc_local_stage_buffers(self, num_stage: int, nt: int):
        delta_x_local_stage = wp.zeros(
            shape=(num_stage, nt), dtype=vec12, device=self.device
        )
        delta_lambda_local_stage = wp.zeros(
            shape=(num_stage, nt), dtype=vec2f, device=self.device
        )
        return delta_x_local_stage, delta_lambda_local_stage

    def solve_all_constraints_local(
        self,
        topo,
        x_old,
        Dm_inv_all,
        k_inv,
        lambda_old,
        gamma_,
        compliance,
        delta_x_local,
        delta_lambda_local,
    ):
        wp.launch(
            kernel=solve_constraints_local_kernel,
            dim=topo.shape[0],
            inputs=[
                topo,
                x_old,
                Dm_inv_all,
                k_inv,
                lambda_old,
                gamma_,
                compliance,
                delta_x_local,
                delta_lambda_local,
                self.eps,
            ],
            device=self.device,
        )

    def solve_elastic_force(
            self,
            topo,
            x_old,
            Dm_inv_all,
            gamma_,
            compliance,
            compliance_modulation,
            local_elastic_force):
        wp.launch(
            kernel=solve_local_elastic_force,
            dim=topo.shape[0],
            inputs=[
                topo,
                x_old,
                Dm_inv_all,
                gamma_,
                compliance,
                compliance_modulation,
                local_elastic_force,
                self.eps,
                self.dt,
            ],
            device=self.device,
        )
        pass