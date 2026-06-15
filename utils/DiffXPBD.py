import os
import numpy as np
import pyglet
from pyglet.math import Vec3 as PyVec3
import warp as wp
import warp.render
from warp._src.render.render_opengl import ShapeInstancer
from typing import List, Optional
import json
import shutil

from utils.MeshLoader import MeshLoader3D_Warp
from utils.KeyPointsBarrycentric import KeyPointsBarrycentric
from utils.PropertyCalculator import PropertyCalculator3D_Warp
from utils.ConstraintSolver import (
    EnergyConstraintSolver3D_Warp,
    vec12,
    vec45f,
    vec2f,
)
from utils.AdamOptimizer import AdamOptimizer

wp.config.verify_autograd_array_access = True


# =========================================================
# Computation Kernels
# =========================================================
@wp.kernel
def clear_mat22_kernel(arr: wp.array(dtype=wp.mat22)):
    i = wp.tid()
    arr[i] = wp.mat22(0.0, 0.0, 0.0, 0.0)

@wp.kernel
def clear_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    arr[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def clear_vec2_kernel(arr: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    arr[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def clear_vec12_kernel(arr: wp.array(dtype=vec12)):
    i = wp.tid()
    arr[i] = vec12()


@wp.kernel
def clear_scalar_kernel(arr: wp.array(dtype=wp.float32)):
    i = wp.tid()
    arr[i] = 0.0

@wp.kernel
def copy_vec3_kernel(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_vec2_kernel(src: wp.array(dtype=wp.vec2), dst: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    dst[i] = src[i]

@wp.kernel
def fill_gravity_kernel(
    out_force: wp.array(dtype=wp.vec3),
    gx: float,
    gy: float,
    gz: float,
    mass_total: float,
    nv: int,
):
    i = wp.tid()
    scale = mass_total / float(nv)
    out_force[i] = wp.vec3(gx * scale, gy * scale, gz * scale)

@wp.kernel
def update_youngs_from_log_kernel(
    youngs: wp.array(dtype=wp.float32),
    youngs_log: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    youngs[tid] = wp.exp(youngs_log[tid])

@wp.kernel
def update_gamma_from_youngs_kernel(
    youngs: wp.array(dtype=wp.float32),
    gamma: wp.array(dtype=wp.float32),
    poisson_ratio: float,
):
    E = youngs[0]
    nu = poisson_ratio

    lambda_lame = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_lame = E / (2.0 * (1.0 + nu))
    gamma[0] = 1.0 + mu_lame / lambda_lame


@wp.kernel
def update_k_inv_from_youngs_kernel(
    youngs: wp.array(dtype=wp.float32),
    fixed_mask: wp.array(dtype=wp.int32),
    k_inv: wp.array(dtype=wp.float32),
    poisson_ratio: float,
    dt: float,
    avg_edge: float,
    selected_vid: int,
    b_amp: wp.array(dtype=wp.float32),
    mass_vertex: wp.array(dtype=wp.float32)
):
    v = wp.tid()
    E = youngs[0]
    nu = poisson_ratio
    damping = (4.0 * dt * E * b_amp[0]) / ((1.0 + nu) * avg_edge * avg_edge)
    k = damping * dt + mass_vertex[v]

    if fixed_mask[v] == 1 or v == selected_vid:
        k_inv[v] = 1.0e-8
    else:
        k_inv[v] = 1.0 / k


@wp.kernel
def update_compliance_from_youngs_kernel(
    youngs: wp.array(dtype=wp.float32),
    tet_volume: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.mat22),
    poisson_ratio: float,
    dt: float,
    compliance_modulation: float,
):
    t = wp.tid()
    E = youngs[0]
    nu = poisson_ratio

    lambda_lame = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_lame = E / (2.0 * (1.0 + nu))

    Hc = (1.0 / (lambda_lame * tet_volume[t] * dt * dt)) * compliance_modulation
    Dc = (1.0 / (mu_lame * tet_volume[t] * dt * dt)) * compliance_modulation
    compliance[t] = wp.mat22(Hc, 0.0, 0.0, Dc)

@wp.kernel
def update_external_force_kernel(
    gravity_force: wp.array(dtype=wp.vec3),
    applied_mask: wp.array(dtype=wp.int32),
    applied_force_per_vertex: wp.array(dtype=wp.vec3f),
    external_force: wp.array(dtype=wp.vec3),
    applied_force_amp: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    f = gravity_force[i]
    if applied_mask[i] == 1:
        f = f + applied_force_per_vertex[0] * applied_force_amp[0]
    external_force[i] = f


@wp.kernel
def update_external_part_from_k_inv_kernel(
    external_force: wp.array(dtype=wp.vec3),
    k_inv: wp.array(dtype=wp.float32),
    external_part: wp.array(dtype=wp.vec3),
    dt: float,
    mass_vertex: wp.array(dtype=wp.float32),
    velocity: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    external_part[i] = k_inv[i] * (mass_vertex[i] * dt * velocity[i] + external_force[i] * dt * dt)


@wp.kernel
def predictor_step_forward_kernel(
    x_state: wp.array(dtype=wp.vec3),
    external_part: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x_state[i] = x_state[i] + external_part[i]


@wp.kernel
def predictor_step_tape_kernel(
    x_state: wp.array(dtype=wp.vec3),
    external_part: wp.array(dtype=wp.vec3),
    x_stage0: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x_stage0[i] = x_state[i] + external_part[i]


@wp.kernel
def update_lambda_from_local_kernel(
    lambda_prev: wp.array(dtype=wp.vec2),
    delta_lambda_local: wp.array(dtype=wp.vec2),
    lambda_next: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    lambda_next[i] = lambda_prev[i] + delta_lambda_local[i]


@wp.kernel
def accumulate_global_local_updates_forward_kernel(
    topo: wp.array(dtype=wp.vec4i),
    vertex_count: wp.array(dtype=wp.int32),
    delta_x_local: wp.array(dtype=vec12),
    x_state: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    tet = topo[t]
    dx = delta_x_local[t]

    id0 = tet[0]
    id1 = tet[1]
    id2 = tet[2]
    id3 = tet[3]

    c0 = wp.vec3(dx[0], dx[1], dx[2]) / float(vertex_count[id0])
    c1 = wp.vec3(dx[3], dx[4], dx[5]) / float(vertex_count[id1])
    c2 = wp.vec3(dx[6], dx[7], dx[8]) / float(vertex_count[id2])
    c3 = wp.vec3(dx[9], dx[10], dx[11]) / float(vertex_count[id3])

    wp.atomic_add(x_state, id0, c0)
    wp.atomic_add(x_state, id1, c1)
    wp.atomic_add(x_state, id2, c2)
    wp.atomic_add(x_state, id3, c3)


@wp.kernel
def gather_next_x_from_local_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    vertex_tet_offsets: wp.array(dtype=wp.int32),
    vertex_tet_ids: wp.array(dtype=wp.int32),
    vertex_local_slot: wp.array(dtype=wp.int32),
    vertex_count: wp.array(dtype=wp.int32),
    delta_x_local: wp.array(dtype=vec12),
    x_next: wp.array(dtype=wp.vec3),):
    v = wp.tid()
    accum = x_prev[v]

    start = vertex_tet_offsets[v]
    end = vertex_tet_offsets[v + 1]
    inv_count = 1.0 / float(vertex_count[v])

    for p in range(start, end):
        t = vertex_tet_ids[p]
        slot = vertex_local_slot[p]
        dx = delta_x_local[t]
        base = 3 * slot
        accum = accum + wp.vec3(dx[base + 0], dx[base + 1], dx[base + 2]) * inv_count

    x_next[v] = accum


@wp.kernel
def calculate_velocity(velocity: wp.array(dtype=wp.vec3), 
                       x_prev: wp.array(dtype=wp.vec3), 
                       x_curr: wp.array(dtype=wp.vec3), 
                       dt: float):
    i = wp.tid()
    velocity[i] = (x_curr[i] - x_prev[i]) / dt


@wp.kernel
def gather_elastic_force_from_local_kernel(
    free_global_elastic_force: wp.array(dtype=wp.vec3),
    fix_global_elastic_force: wp.array(dtype=wp.vec3),
    total_global_elastic_force: wp.array(dtype=wp.vec3),
    vertex_tet_offsets: wp.array(dtype=wp.int32),
    vertex_tet_ids: wp.array(dtype=wp.int32),
    vertex_local_slot: wp.array(dtype=wp.int32),
    vertex_count: wp.array(dtype=wp.int32),
    elastic_force_local: wp.array(dtype=vec12),
    free_mask: wp.array(dtype=wp.int32)
):
    v = wp.tid()

    v_force = wp.vec3(0.0, 0.0, 0.0)

    start = vertex_tet_offsets[v]
    end = vertex_tet_offsets[v + 1]
    inv_count = 1.0 / float(vertex_count[v])

    for p in range(start, end):
        t = vertex_tet_ids[p]
        slot = vertex_local_slot[p]
        df = elastic_force_local[t]
        base = 3 * slot
        v_force = v_force + wp.vec3(df[base + 0], df[base + 1], df[base + 2])

    total_global_elastic_force[v] = v_force
    if free_mask[v] == 0:
        fix_global_elastic_force[v] = v_force
    else:
        free_global_elastic_force[v] = v_force

@wp.kernel
def compute_loss_all_pts_kernel(
    x_final: wp.array(dtype=wp.vec3),
    x_target: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    v = wp.tid()
    d = x_final[v] - x_target[v]
    wp.atomic_add(loss, 0, wp.dot(d, d))

@wp.kernel
def compute_kps_loss_kernel(
    kps_final: wp.array(dtype=wp.vec3f),
    kps_target: wp.array(dtype=wp.vec3f),
    loss: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    d = kps_final[k] - kps_target[k]
    wp.atomic_add(loss, 0, wp.dot(d, d))

@wp.kernel
def set_single_vertex_position(x_state: wp.array(dtype=wp.vec3), vid: wp.int32, px: wp.float32, py: wp.float32, pz: wp.float32):
    x_state[vid] = wp.vec3(px, py, pz)

@wp.kernel
def keypoints_update(x_state: wp.array(dtype=wp.vec3), 
                     topo: wp.array(dtype=wp.vec4i), 
                     keypoints_tet: wp.array(dtype=wp.int32), 
                     keypoints_barycentric: wp.array(dtype=wp.vec4), 
                     keypoints_pos: wp.array(dtype=wp.vec3f)):
    k = wp.tid()
    tet_id = keypoints_tet[k]
    v1 = x_state[topo[tet_id][0]]
    v2 = x_state[topo[tet_id][1]]
    v3 = x_state[topo[tet_id][2]]
    v4 = x_state[topo[tet_id][3]]
    w = keypoints_barycentric[k]
    keypoints_pos[k] = v1 * w[0] + v2 * w[1] + v3 * w[2] + v4 * w[3]


class DiffXPBDTapeFramework3D_Warp:
    def __init__(
        self,
        trajectory_type: str = "kps", ## Trajectory type for loss calculation, "kps" or "mesh_pts"
        optimize_type: str = "sim2sim", ## Optimization type, "sim2sim" or "sim2real"
        mesh_path: str = None,
        real_target_kps_path: str = None,
        sim_target_npz: Optional[str] = None,
        target_unit: str = "mm",
        dt: float = 1.e-2,
        gravity_vec: tuple[float, float, float] = (0.0, -9.81, 0.0),
        mass_total: float = 1.0,
        poisson_ratio: float = 0.48,
        compliance_modulation: float = 1.e-2,

        fixed_center: List[tuple[float, float, float]] = None,
        fixed_rotation: List[tuple[float, float, float]] = None,
        fixed_extents: List[tuple[float, float, float]] = None,

        applied_center: List[tuple[float, float, float]] = None,
        applied_extents: List[tuple[float, float, float]] = None,
        show_force_arrow: bool = False,
        constant_applied_force: tuple[float, float, float] = None,
        series_force_path: Optional[str] = None,
        series_force_mode: Optional[bool] = False,
        
        youngs_init: float = 10 * 1e6,
        force_amplification: float = 1.e0,
        damping_amplification: float = 0.1,
        mass_amplification: float = 1.0,
        sweep_count: int = 20,

        position_effector_path: str = None,

        camera_pos: tuple[float, float, float] = (0., 180., 45.2006),
        camera_front: tuple[float, float, float] = (0.0, -1.0, 0.0),
        camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        camera_look_at: tuple[float, float, float] = (0., 0., 45.2006),
        background_color: tuple[float, float, float] = (1., 1., 1.),
        mesh_color: tuple[float, float, float] = (0.53, 0.81, 0.92),
        text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
        axial_scale: float = 10.0,

        device: Optional[str] = None,
    ):
        self.device = device if device is not None else wp.get_preferred_device()

        self.trajctory_type = trajectory_type
        self.optimize_type = optimize_type
        if self.trajctory_type not in ["kps", "mesh_pts"]:
            raise ValueError("trajectory_type must be 'kps' or 'mesh_pts'.")
        elif self.optimize_type not in ["sim2sim", "sim2real"]:
            raise ValueError("optimize_type must be 'sim2sim' or 'sim2real'.")
        else:
            if self.trajctory_type == "kps":
                if self.optimize_type == "sim2sim":
                    self.target_init = self._initialize_kps_target_sim2sim
                    self.target_path = sim_target_npz
                elif self.optimize_type == "sim2real":
                    self.target_init = self._initialize_kps_target_sim2real
                    self.target_path = real_target_kps_path
            else:
                self.target_init = self._initialize_target
                self.target_path = sim_target_npz

        if target_unit == "mm":
            self.scale_modulation = 1.0
            self.force_modulation = 1.0e3
            self.stress_modulation = 1.0e-3
            self.acceleration_modulation = 1.0e3
        elif target_unit == "m":
            self.scale_modulation = 1.0
            self.force_modulation = 1.0
            self.stress_modulation = 1.0
            self.acceleration_modulation = 1.0
        else:
            raise ValueError("target_unit must be 'mm' or 'm'.")

        self.dt = float(dt)
        self.gravity_vec = np.asarray(gravity_vec, dtype=np.float32) * self.acceleration_modulation
        self.mass_amplification = float(mass_amplification)
        self.mass_total = float(mass_total) * self.mass_amplification
        self.poisson_ratio = float(poisson_ratio)
        self.compliance_modulation = float(compliance_modulation)
        self.sweep_count = int(sweep_count)
        self.num_steps = self.sweep_count + 1

        ##### Load Gripper Mesh #####
        self.mesh = MeshLoader3D_Warp(device=self.device)
        self.mesh.load_gmsh(mesh_path, target_unit=target_unit)
        self.boundary_indices = self.mesh.get_oriented_boundary_faces_numpy().astype(np.int32)
        self.gripper_edge_indices = self.mesh.E_np.reshape(-1).astype(np.int32)

        self.mass_per_vertex = self.mass_total / float(self.mesh.nv)
        self.mass_vertex_np = np.full(self.mesh.nv, self.mass_per_vertex, dtype=np.float32)
        self.mass_per_vertex_field = wp.array(self.mass_vertex_np, dtype=wp.float32, device=self.device, requires_grad=True)

        self.pc = PropertyCalculator3D_Warp(
            self.mesh,
            device=self.device,
            compliance_modulation=self.compliance_modulation,
        )

        self.avg_edge = self.pc.avg_len
        self.drag_pick_radius = max(2.0 * float(self.avg_edge), 1e-3)

        ##### Load Gripper Mesh #####
        self.keypointmapper = KeyPointsBarrycentric(position_effector_path, self.mesh, device)
        self.keypoints_pos = wp.array(self.keypointmapper.keypoints, dtype=wp.vec3, device=self.device)

        ########## Set up Boundary Conditions ##########
        
        ##### Fixed Boundary Conditions #####
        _, self.fixed_idx = self.pc.points_in_obb_3d_batch(
            np.asarray(fixed_center, dtype=np.float32).reshape(-1, 3),
            np.asarray(fixed_rotation, dtype=np.float32).reshape(-1, 3),
            np.asarray(fixed_extents, dtype=np.float32).reshape(-1, 3),
        )
        fixed_mask_np = np.zeros(self.mesh.nv, dtype=np.int32)
        fixed_mask_np[self.fixed_idx] = 1
        fixed_mask_np = np.ascontiguousarray(fixed_mask_np)
        self.fixed_mask = wp.array(fixed_mask_np, dtype=wp.int32, device=self.device)

        ##### Applied Boundary Conditions #####
        _, self.applied_idx = self.pc.points_in_aabb_3d_batch(
            np.asarray(applied_center, dtype=np.float32),
            np.asarray(applied_extents, dtype=np.float32),
        )
        self.applied_center_height = applied_center[0][-1]
        self.applied_idx = self.applied_idx.astype(np.int32)
        applied_mask_np = np.zeros(self.mesh.nv, dtype=np.int32)
        applied_mask_np[self.applied_idx] = 1
        applied_mask_np = np.ascontiguousarray(applied_mask_np)
        self.applied_mask = wp.array(applied_mask_np, dtype=wp.int32, device=self.device)
        self.applied_force_np = np.asarray(constant_applied_force, dtype=np.float32) * self.force_modulation

        # Visualization settings
        self.fixed_idx_np = np.ascontiguousarray(np.array(self.fixed_idx, dtype=np.int32))
        self.applied_idx_np = np.ascontiguousarray(np.array(self.applied_idx, dtype=np.int32))
        mask = np.ones(self.mesh.nv, dtype=bool)

        mask[self.fixed_idx_np] = False
        mask[self.applied_idx_np] = False

        self.free_idx_np = np.ascontiguousarray(np.where(mask)[0].astype(np.int32))
        free_mask = np.zeros(self.mesh.nv, dtype=np.int32)
        free_mask[self.free_idx_np] = 1
        self.free_mask_wp = wp.array(free_mask, dtype=wp.int32, device=self.device)

        ## Model Parameters
        self.youngs_np = float(youngs_init) * self.stress_modulation
        self.youngs_log_np = np.log(self.youngs_np)

        self.youngs_log = wp.array([self.youngs_log_np], dtype=wp.float32, device=self.device, requires_grad=True)
        # self.youngs = wp.array([self.youngs_np], dtype=wp.float32, device=self.device, requires_grad=True)
        self.youngs = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self.update_youngs_from_log(self.youngs_log, self.youngs)
        self.youngs_np = float(self.youngs.numpy()[0])

        self.damping_amp_np = float(damping_amplification)
        self.damping_amp = wp.array([self.damping_amp_np], dtype=wp.float32, device=self.device, requires_grad=True)

        self.show_force_arrow = show_force_arrow
        self.series_force_path = series_force_path
        self.series_force_mode = series_force_mode
        if optimize_type == "sim2real":
            self.series_force_mode = True
        self.series_force_amp_np = force_amplification

        if self.series_force_path is not None:
            self._initialize_series_forces(self.series_force_path, max(len(self.applied_idx), 1))

        self.per_vertex_force = self.applied_force_np / max(len(self.applied_idx), 1)

        self.applied_force_field = wp.array([wp.vec3(*self.per_vertex_force.tolist())], dtype=wp.vec3f, device=self.device, requires_grad=True)
        self.gamma = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        self.k_inv = wp.zeros(self.mesh.nv, dtype=wp.float32, device=self.device, requires_grad=True)
        self.compliance = wp.zeros(self.mesh.nt, dtype=wp.mat22, device=self.device, requires_grad=True)

        self.gravity_vec_field = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device)
        self.external_force = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.external_part = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=True)

        self.x_state_prev = wp.clone(self.mesh.X)
        self.x_state_curr = wp.clone(self.mesh.X)
        self.x_velocity = wp.zeros_like(self.mesh.X)
        self.x_np = None
        self.lambda_state = wp.zeros(self.mesh.nt, dtype=wp.vec2, device=self.device)

        self.delta_x_local = wp.zeros(self.mesh.nt, dtype=vec12, device=self.device)
        self.delta_lambda_local = wp.zeros(self.mesh.nt, dtype=wp.vec2, device=self.device)

        ## Elastic Force
        self.free_node_elastic_force = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=False)
        self.fix_node_elastic_force = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=False)
        self.total_node_elastic_force = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=False)
        self.local_elastic_force = wp.zeros(self.mesh.nt, dtype=vec12, device=self.device)
        self.total_free_node_force = np.zeros(3, dtype=np.float32)
        # self.total_force_nm = np.zeros(3, dtype=np.float32)

        self.solver = EnergyConstraintSolver3D_Warp(device=self.device, dt=self.dt)
        self._build_vertex_tet_adjacency()
        self._init_gravity()
        self.target_trajectory = None

        self.loss_hist = []
        self.grad_E_log_hist = []
        self.grad_F_hist = []
        self.grad_F_amp_hist = []
        self.grad_damping_amp_hist = []

        self.x_hist = []
        self.kps_hist = []

        # Camera Settings
        self.camera_pos = np.array(camera_pos)
        self.camera_front = np.array(camera_front)
        self.camera_up = np.array(camera_up)
        self.camera_look_at = np.array(camera_look_at)
        self.orbit_radius = abs((self.camera_pos - self.camera_look_at)[1])
        self.background_color = np.array(background_color)

        self.init_camera_pos = np.array(camera_pos)
        self.init_camera_front = np.array(camera_front)
        self.init_camera_up = np.array(camera_up)
        self.init_camera_look_at = np.array(camera_look_at)

        self.mesh_color = np.array(mesh_color)
        self.text_color = np.array(text_color, dtype=np.uint8)

        self.axial_scale = axial_scale

        # OpenGL Renderer
        self.rotational_angle = 0.025
        ## Drag Helpers
        self.non_draggable_mask_np = np.zeros(self.mesh.nv, dtype=bool)
        self.non_draggable_mask_np[self.fixed_idx_np] = True
        self.selected_vid = -1
        self.ray_o = None   
        self.ray_d = None

        # USD rendering / export
        self.usd_renderer = None
        self.usd_path = None
        self.usd_mesh_name = "deformable_mesh"
        self.usd_scale = 1.0

        self.target_init(self.target_path)

    # -----------------------------------------------------
    # adjacency for tape-safe gather
    # -----------------------------------------------------
    def _build_vertex_tet_adjacency(self):
        topo = self.mesh.Topo_np
        nv = self.mesh.nv
        incident = [[] for _ in range(nv)]
        for t in range(self.mesh.nt):
            tet = topo[t]
            for slot in range(4):
                incident[int(tet[slot])].append((t, slot))

        offsets = np.zeros(nv + 1, dtype=np.int32)
        total = 0
        for v in range(nv):
            offsets[v] = total
            total += len(incident[v])
        offsets[nv] = total

        tet_ids = np.zeros(total, dtype=np.int32)
        local_slot = np.zeros(total, dtype=np.int32)
        p = 0
        for v in range(nv):
            for t, slot in incident[v]:
                tet_ids[p] = t
                local_slot[p] = slot
                p += 1

        self.vertex_tet_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.vertex_tet_ids = wp.array(tet_ids, dtype=wp.int32, device=self.device)
        self.vertex_local_slot = wp.array(local_slot, dtype=wp.int32, device=self.device)

    def _init_gravity(self):
        wp.launch(
            fill_gravity_kernel,
            dim=self.mesh.nv,
            inputs=[
                self.gravity_vec_field,
                float(self.gravity_vec[0]),
                float(self.gravity_vec[1]),
                float(self.gravity_vec[2]),
                self.mass_total,
                self.mesh.nv,
            ],
            device=self.device,
        )

    def _initialize_target(self, target_npz: str):
        pos = np.load(target_npz)["position"]
        if pos.shape[1] != self.mesh.nv or pos.shape[2] != 3:
            raise ValueError(
                f"target_npz['position'] shape should be (T, {self.mesh.nv}, 3), got {pos.shape}"
            )
        self.target_trajectory = [
            wp.array(pos[k].astype(np.float32), dtype=wp.vec3, device=self.device)
            for k in range(pos.shape[0])
        ]

    def _initialize_kps_target_sim2sim(self, target_npz: str):
            kp_pos = np.array(np.load(target_npz)["keypoints"]).reshape(-1, self.keypointmapper.kp_num * 3)
            self.target_trajectory = [
                wp.array(kp_pos[k].reshape(-1, 3).astype(np.float32), dtype=wp.vec3f, device=self.device)
                for k in range(kp_pos.shape[0])
            ]

    def _initialize_kps_target_sim2real(self, target_kps_path: str):
        data = np.loadtxt(target_kps_path, delimiter=',')
        kp_pos = data[:, :-1].reshape(-1, self.keypointmapper.kp_num * 3)
        self.target_trajectory = [
            wp.array(kp_pos[k].reshape(-1, 3).astype(np.float32), dtype=wp.vec3f, device=self.device)
            for k in range(kp_pos.shape[0])
        ]


    def _initialize_series_forces(self, series_force_path: str, vertices_num: int):
        data = np.loadtxt(series_force_path, delimiter=',')
        applied_vertices_forces = (data[:, :-1].reshape(-1, 3) * self.force_modulation) / vertices_num
        self.applied_vertices_forces = [
            wp.array(applied_vertices_forces[k].astype(np.float32), dtype=wp.vec3f, device=self.device, requires_grad=True)
            for k in range(applied_vertices_forces.shape[0])
        ]
    # -----------------------------------------------------
    # parameter updates
    # -----------------------------------------------------
    def update_gamma_from_youngs(self, gamma):
        wp.launch(update_gamma_from_youngs_kernel, 
                  dim=1, 
                  inputs=[self.youngs, gamma, self.poisson_ratio], 
                  device=self.device)

    def update_k_inv_from_youngs(self, k_inv, damping_amp, mass_vertex):
        wp.launch(
            update_k_inv_from_youngs_kernel,
            dim=self.mesh.nv,
            inputs=[self.youngs, 
                    self.fixed_mask, 
                    k_inv, 
                    self.poisson_ratio, 
                    self.dt, 
                    float(self.avg_edge), 
                    int(self.selected_vid), 
                    damping_amp,
                    mass_vertex],
            device=self.device,
        )

    def update_compliance_from_youngs(self, compliance):
        wp.launch(
            update_compliance_from_youngs_kernel,
            dim=self.mesh.nt,
            inputs=[self.youngs, self.pc.tet_volume, compliance, self.poisson_ratio, self.dt, self.compliance_modulation],
            device=self.device,
        )

    def update_external_force(self, external_force, applied_force_field, applied_force_amp):
        wp.launch(
            update_external_force_kernel,
            dim=self.mesh.nv,
            inputs=[self.gravity_vec_field, self.applied_mask, applied_force_field, external_force, applied_force_amp],
            device=self.device,
        )

    def update_external_part_from_youngs(self, external_force, k_inv, external_part, mass_vertex, velocity):
        wp.launch(
            update_external_part_from_k_inv_kernel,
            dim=self.mesh.nv,
            inputs=[external_force, 
                    k_inv, 
                    external_part, 
                    self.dt,
                    mass_vertex,
                    velocity
                    ],
            device=self.device,
        )

    def calculate_local_elastic_force(self, gamma, compliance):
        self.solver.solve_elastic_force(
            topo=self.mesh.Topo,
            x_old=self.x_state_curr,
            Dm_inv_all=self.pc.Dm_inv,
            gamma_=gamma,
            compliance=compliance,
            compliance_modulation=self.compliance_modulation,
            local_elastic_force=self.local_elastic_force,
        )

    def calculate_current_velocity(self, velocity, x_prev, x_curr):
        wp.launch(
            calculate_velocity,
            dim=self.mesh.nv,
            inputs=[velocity, x_prev, x_curr, self.dt],
            device=self.device,
        )

    def calculate_global_elastic_force_wp(self,):
        wp.launch(
            gather_elastic_force_from_local_kernel,
            dim=self.mesh.nv,
            inputs=[
                self.free_node_elastic_force,
                self.fix_node_elastic_force,
                self.total_node_elastic_force,
                self.vertex_tet_offsets,
                self.vertex_tet_ids,
                self.vertex_local_slot,
                self.pc.vertex_count,
                self.local_elastic_force,
                self.free_mask_wp,
            ],
            device=self.device,
        )
        self.total_free_node_force = self.free_node_elastic_force.numpy().sum(axis=0).reshape(3,) / self.force_modulation

        self.total_fix_node_force = self.fix_node_elastic_force.numpy().sum(axis=0).reshape(3,) / self.force_modulation

        self.total_node_force = self.total_node_elastic_force.numpy().sum(axis=0).reshape(3,) / self.force_modulation

        self.applied_node_force = self.total_node_elastic_force.numpy()[self.applied_idx_np].sum(axis=0).reshape(3,) / self.force_modulation
        # total_force_min = max(min(np.abs(self.total_free_node_force)), 1.0)
        # self.total_free_node_force = self.total_free_node_force / total_force_min
        # self.total_force_nm = self.total_force/np.linalg.norm(self.total_force)

    # -----------------------------------------------------
    # forward pass
    # -----------------------------------------------------
    def reset_lambda_states(self):
        wp.launch(clear_vec2_kernel, dim=self.mesh.nt, inputs=[self.lambda_state], device=self.device)

    def clear_new_state(self):
        wp.launch(clear_vec12_kernel, dim=self.mesh.nt, inputs=[self.delta_x_local], device=self.device)
        wp.launch(clear_vec2_kernel, dim=self.mesh.nt, inputs=[self.delta_lambda_local], device=self.device)

    def predictor_step_gui(self):
        wp.launch(predictor_step_forward_kernel, dim=self.mesh.nv, inputs=[self.x_state_curr, self.external_part], device=self.device)

    def solve_all_constraints_gui(self):
        self.solver.solve_all_constraints_local(
            topo=self.mesh.Topo,
            x_old=self.x_state_curr,
            Dm_inv_all=self.pc.Dm_inv,
            k_inv=self.k_inv,
            lambda_old=self.lambda_state,
            gamma_=self.gamma,
            compliance=self.compliance,
            delta_x_local=self.delta_x_local,
            delta_lambda_local=self.delta_lambda_local,
        )
        wp.launch(
            update_lambda_from_local_kernel,
            dim=self.mesh.nt,
            inputs=[self.lambda_state, self.delta_lambda_local, self.lambda_state],
            device=self.device,
        )
        wp.launch(
            accumulate_global_local_updates_forward_kernel,
            dim=self.mesh.nt,
            inputs=[self.mesh.Topo, self.pc.vertex_count, self.delta_x_local, self.x_state_curr],
            device=self.device,
        )

    def update_keypoints_gui(self):
        wp.launch(
            keypoints_update,
            dim=self.keypointmapper.kp_num,
            inputs=[self.x_state_curr, self.mesh.Topo, self.keypointmapper.keypoints_tet_wp, self.keypointmapper.keypoints_barycentric_wp, self.keypoints_pos],
            device=self.device,
        )

    def clear_loss(self):
        wp.launch(clear_scalar_kernel, dim=1, inputs=[self.loss], device=self.device)

    def accumulate_loss_all_pts_gui(self, step: int):
        if self.target_trajectory is None:
            raise RuntimeError("x_target is not initialized. Pass target_npz to the constructor.")
        wp.launch(
            compute_loss_all_pts_kernel,
            dim=self.mesh.nv,
            inputs=[self.x_state_curr, self.target_trajectory[step], self.loss],
            device=self.device,
        )

    def accumulate_kps_loss(self, step: int, keypoints_pos):
        if self.target_trajectory is None:
            raise RuntimeError("x_target is not initialized. Pass target_npz to the constructor.")
        wp.launch(
            compute_kps_loss_kernel,
            dim=self.keypointmapper.kp_num,
            inputs=[keypoints_pos, self.target_trajectory[step], self.loss],
            device=self.device,
        )

    def one_forward_step(self, step: int):
        ## Store previous state
        wp.copy(self.x_state_prev, self.x_state_curr)

        self.free_node_elastic_force.zero_()
        self.fix_node_elastic_force.zero_()
        self.total_node_elastic_force.zero_()
        if self.series_force_mode is False:
            self.applied_force_field = wp.array([wp.vec3(*self.per_vertex_force.tolist())], dtype=wp.vec3f, device=self.device, requires_grad=True)
        elif step < len(self.applied_vertices_forces):
            self.applied_force_field = self.applied_vertices_forces[step]
        elif step >= len(self.applied_vertices_forces):
            self.applied_force_field = wp.array([[0,0,0]], dtype=wp.vec3f, device=self.device, requires_grad=True)

        self.applied_force_amp = wp.array([self.series_force_amp_np], dtype=wp.float32, device=self.device, requires_grad=False)

        self.update_gamma_from_youngs(self.gamma)
        self.update_k_inv_from_youngs(self.k_inv, self.damping_amp, self.mass_per_vertex_field)
        self.update_compliance_from_youngs(self.compliance)
        self.update_external_force(self.external_force, self.applied_force_field, self.applied_force_amp)
        self.update_external_part_from_youngs(self.external_force, self.k_inv, self.external_part, self.mass_per_vertex_field, self.x_velocity)
        self.predictor_step_gui()
        self.reset_lambda_states()

        for sweep in range(self.sweep_count):
            self.clear_new_state()
            self.solve_all_constraints_gui()

        self.calculate_current_velocity(self.x_velocity, self.x_state_prev, self.x_state_curr)
        self.update_keypoints_gui()

        if self.target_trajectory is not None:
            self.clear_loss()
            if step >= len(self.target_trajectory):
                step = -1
            if self.trajctory_type == "kps":
                self.accumulate_kps_loss(step, self.keypoints_pos)
            else:
                self.accumulate_loss_all_pts_gui(step)

        self.calculate_local_elastic_force(self.gamma, self.compliance)
        self.calculate_global_elastic_force_wp()

        loss = float(self.loss.numpy()[0])
        return loss

    # -----------------------------------------------------
    # tape path
    # -----------------------------------------------------

    def update_youngs_from_log(self, youngs_log, youngs):
        wp.launch(update_youngs_from_log_kernel, 
                  dim=1, 
                  inputs=[youngs, youngs_log], 
                  device=self.device)

    def predictor_step_tape(self, external_part, x_tape):
        wp.launch(
            predictor_step_tape_kernel,
            dim=self.mesh.nv,
            inputs=[self.x_state_curr, external_part, x_tape[0]],
            device=self.device,
        )

    def solve_one_constraint_sweep_tape(self, stage: int, x_tape, b_inv, lambda_tape, gamma, compliance, delta_x_local_tape, delta_lambda_local_tape):
        self.solver.solve_all_constraints_local(
            topo=self.mesh.Topo,
            x_old=x_tape[stage],
            Dm_inv_all=self.pc.Dm_inv,
            k_inv=b_inv,
            lambda_old=lambda_tape[stage],
            gamma_=gamma,
            compliance=compliance,
            delta_x_local=delta_x_local_tape[stage],
            delta_lambda_local=delta_lambda_local_tape[stage],
        )

    def accumulate_global_local_updates_tape(self, stage: int, x_tape, lambda_tape, delta_x_local_tape, delta_lambda_local_tape):
        wp.launch(
            gather_next_x_from_local_kernel,
            dim=self.mesh.nv,
            inputs=[
                x_tape[stage],
                self.vertex_tet_offsets,
                self.vertex_tet_ids,
                self.vertex_local_slot,
                self.pc.vertex_count,
                delta_x_local_tape[stage],
                x_tape[stage + 1],
            ],
            device=self.device,
        )
        wp.launch(
            update_lambda_from_local_kernel,
            dim=self.mesh.nt,
            inputs=[lambda_tape[stage], delta_lambda_local_tape[stage], lambda_tape[stage + 1]],
            device=self.device,
        )

    def update_keypoints_tape(self, stage, keypoints_pos):
        wp.launch(
            keypoints_update,
            dim=self.keypointmapper.kp_num,
            inputs=[self.x_tape[stage], self.mesh.Topo, self.keypointmapper.keypoints_tet_wp, self.keypointmapper.keypoints_barycentric_wp, keypoints_pos],
            device=self.device,
        )

    def accumulate_loss_all_pts(self, stage: int, step: int):
        if self.target_trajectory is None:
            raise RuntimeError("x_target is not initialized. Pass target_npz to the constructor.")
        wp.launch(
            compute_loss_all_pts_kernel,
            dim=self.mesh.nv,
            inputs=[self.x_tape[stage], self.target_trajectory[step], self.loss],
            device=self.device,
        )

    def run_with_tape(self, step: int):
        tape = wp.Tape()
        self.free_node_elastic_force.zero_()
        self.fix_node_elastic_force.zero_()
        self.total_node_elastic_force.zero_()
        if self.series_force_mode is False or self.series_force_path is None:
            self.applied_force_field = wp.array([wp.vec3(*self.per_vertex_force.tolist())], dtype=wp.vec3f, device=self.device, requires_grad=True)
        elif step < len(self.applied_vertices_forces):
            self.applied_force_field = self.applied_vertices_forces[step]
        else:
            self.applied_force_field = wp.array([[0,0,0]], dtype=wp.vec3f, device=self.device, requires_grad=True)

        self.youngs_log = wp.array([self.youngs_log_np], dtype=wp.float32, device=self.device, requires_grad=True)
        # self.youngs = wp.array([self.youngs_np], dtype=wp.float32, device=self.device, requires_grad=True)
        self.youngs = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self.damping_amp = wp.array([self.damping_amp_np], dtype=wp.float32, device=self.device, requires_grad=True)
        self.x_tape = [wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=True) for _ in range(self.num_steps)]
        lambda_tape = [wp.zeros(self.mesh.nt, dtype=wp.vec2, device=self.device, requires_grad=True) for _ in range(self.num_steps)]
        delta_x_local_tape = [wp.zeros(self.mesh.nt, dtype=vec12, device=self.device, requires_grad=True) for _ in range(self.sweep_count)]
        delta_lambda_local_tape = [wp.zeros(self.mesh.nt, dtype=wp.vec2, device=self.device, requires_grad=True) for _ in range(self.sweep_count)]
        gamma = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        b_inv = wp.zeros(self.mesh.nv, dtype=wp.float32, device=self.device, requires_grad=True)
        mass_vertex = wp.array([self.mass_per_vertex], dtype=wp.float32, device=self.device, requires_grad=True)
        compliance = wp.zeros(self.mesh.nt, dtype=wp.mat22, device=self.device, requires_grad=True)

        applied_force_amp = wp.array([self.series_force_amp_np], dtype=wp.float32, device=self.device, requires_grad=True)
        external_force = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=True)
        external_part = wp.zeros(self.mesh.nv, dtype=wp.vec3, device=self.device, requires_grad=True)
        keypoints_pos = wp.zeros(self.keypointmapper.kp_num, dtype=wp.vec3f, device=self.device, requires_grad=True)
        with tape:
            self.update_youngs_from_log(self.youngs_log, self.youngs)
            self.update_gamma_from_youngs(gamma)
            self.update_k_inv_from_youngs(b_inv, self.damping_amp, mass_vertex)
            self.update_compliance_from_youngs(compliance)
            self.update_external_force(external_force, self.applied_force_field, applied_force_amp)
            self.update_external_part_from_youngs(external_force, b_inv, external_part)

            self.predictor_step_tape(external_part, self.x_tape)
            stage = 0
            for sweep in range(self.sweep_count):
                self.solve_one_constraint_sweep_tape(stage, self.x_tape, b_inv, lambda_tape, gamma, compliance, delta_x_local_tape, delta_lambda_local_tape)
                self.accumulate_global_local_updates_tape(stage, self.x_tape, lambda_tape, delta_x_local_tape, delta_lambda_local_tape)
                stage += 1

            self.update_keypoints_tape(stage, keypoints_pos)
            self.clear_loss()
            if self.trajctory_type == "kps":
                self.accumulate_kps_loss(step, keypoints_pos)
            else:
                self.accumulate_loss_all_pts(stage, step)
            

        tape.backward(loss=self.loss)
        # tape.visualize(
        #     filename="tape.dot",
        #     array_labels={ self.youngs_log: "E_log", self.applied_force_field: "F", self.loss: "loss"},
        # )

        grad_E_log = float(self.youngs_log.grad.numpy()[0]) if self.youngs_log.grad is not None else 0.0
        grad_F_amp = float(applied_force_amp.grad.numpy()[0]) if applied_force_amp.grad is not None else 0.0
        grad_F = self.applied_force_field.grad.numpy()[0].tolist() if self.applied_force_field.grad is not None else [0.0, 0.0, 0.0]
        grad_damping_amp = float(self.damping_amp.grad.numpy()[0]) if self.damping_amp.grad is not None else 0.0

        loss = float(self.loss.numpy()[0])

        wp.copy(self.x_state_prev, self.x_state_curr)
        wp.copy(self.x_state_curr, self.x_tape[-1])
        wp.copy(self.keypoints_pos, keypoints_pos)
        self.youngs_np = float(self.youngs.numpy()[0])

        # self.calculate_local_elastic_force(gamma, compliance)
        # self.calculate_global_elastic_force_wp()
        return loss, grad_E_log, grad_F, grad_F_amp, grad_damping_amp

    # -----------------------------------------------------
    # utility
    # -----------------------------------------------------
    def copy_state_to_numpy(self):
        return self.x_state_curr.numpy()

    def reset_to_rest(self):
        self.loss_hist = []
        self.grad_E_log_hist = []
        self.grad_F_hist = []
        self.grad_F_amp_hist = []
        self.grad_damping_amp_hist = []
        wp.copy(self.x_state_curr, self.mesh.X)
        wp.launch(clear_vec2_kernel, dim=self.mesh.nt, inputs=[self.lambda_state], device=self.device)

    # -----------------------------------------------------
    # OpenGL visualization
    # -----------------------------------------------------
    def init_opengl_renderer(
        self,
        screen_width: int = 1280,
        screen_height: int = 920,
        fps: int = None,
        interactive_flag: bool = False,
    ):
        if fps is None:
            fps = int(1.0 / (self.dt))

        self.opengl_renderer = warp.render.OpenGLRenderer(
            title = "Hybrid XPBD",
            fps=fps,
            screen_width=screen_width,
            screen_height=screen_height,
            far_plane= 500,
            camera_pos=self.camera_pos,
            camera_front=self.camera_front,
            camera_up=self.camera_up,
            draw_grid=False,
            draw_axis=True,
            background_color=self.background_color,
            enable_backface_culling = False,
            axis_scale=self.axial_scale,
            enable_mouse_interaction = False,
            enable_keyboard_interaction = False,
            vsync=False,
        )

        # self.opengl_renderer.window.set_mouse_visible(False)
        # DIY Key interaction
        self.key_handler = self.opengl_renderer._key_handler
        if interactive_flag:
            self.mouse_pos = np.array([0., 0.])
            self.opengl_renderer.window.push_handlers(on_mouse_motion=self._my_mouse_motion,
                                                    on_mouse_press=self._my_mouse_press,
                                                    on_mouse_drag=self._my_mouse_drag,
                                                    on_mouse_release=self._my_mouse_release,
                                                    on_mouse_scroll=self._my_mouse_scroll,)
            
        self.my_label = pyglet.text.Label(
            '',
            font_name='Arial',
            font_size=12,
            x=10,
            y=10,
            anchor_x='left',
            anchor_y='top',
            color=self.text_color,  # RGBA
            multiline=True,
            width = 500
        )
        self.opengl_renderer.render_2d_callbacks.append(self._draw_my_overlay)

        # Rendering Switches
        self.show_nodes = True
        self.show_fixed_points = True
        self.show_free_points = True
        self.show_applied_points = True

        if self.show_force_arrow and self.show_applied_points is not None and len(self.applied_idx_np) > 0:
            self.arrow_dir = np.asarray([0., 1., 0.], dtype=np.float32)
            vertices, indices = self.opengl_renderer._create_arrow_mesh(
                base_radius=0.02 *  10, base_height=0.85 * self.axial_scale, cap_height=0.15 * self.axial_scale
            )
            self.force_instancer = ShapeInstancer(self.opengl_renderer._shape_shader, wp.get_preferred_device())
            self.force_instancer.register_shape(vertices, indices)
            self.force_instancer.allocate_instances(
                positions=[tuple(self.x_state_curr.numpy()[vid]) for vid in self.applied_idx_np],
                rotations=[self.quat_from_arrow_to_force(self.applied_force_field.numpy()[0]) for _ in range(len(self.applied_idx_np))],
                colors1=[(1.0, 0.0, 0.0) for _ in range(len(self.applied_idx_np))],
                colors2=[(1.0, 0.0, 0.0) for _ in range(len(self.applied_idx_np))],
                scalings=[tuple(self.applied_force_field.numpy()[0]) for _ in range(len(self.applied_idx_np))],
            )
            self.opengl_renderer._shape_instancers["force_arrow"] = self.force_instancer

        self.frame_id = 0

    def render_opengl_frame(self, step):
        if self.opengl_renderer is None:
            raise RuntimeError("OpenGL renderer is not initialized. Call init_opengl_renderer() first.")

        points = self.x_state_curr.numpy()
        # time_in_seconds = float(self.frame_id) / float(self.opengl_renderer.fps)
        
        self.update_render_force_arrows()

        self.opengl_renderer.begin_frame()

        self.opengl_renderer.render_mesh(
            name="deformable_mesh",
            points=points,
            indices=self.boundary_indices,
            colors=self.mesh_color,
            update_topology=(self.frame_id == 0),
            visible=True,
        )

        if self.show_nodes and self.show_fixed_points and len(self.fixed_idx_np) > 0:
            self.opengl_renderer.render_points(
                name="fixed_points",
                points=points[self.fixed_idx_np],
                radius=0.4,
                colors=(1.0, 0.753, 0.796),
                as_spheres=True,
                visible=True,
            )

        if self.show_nodes and self.show_free_points and len(self.free_idx_np) > 0:
            self.opengl_renderer.render_points(
                name="free_points",
                points=points[self.free_idx_np],
                radius=0.25,
                colors=(0.0, 0.0, 1.0),
                as_spheres=True,
                visible=True,
            )

        if self.show_nodes and self.show_applied_points and len(self.applied_idx_np) > 0:
            self.opengl_renderer.render_points(
                name="applied_points",
                points=points[self.applied_idx_np],
                radius=0.4,
                colors=(1.0, 0.0, 0.0),
                as_spheres=True,
                visible=True,
            )

        if self.keypointmapper.keypoints is not None and len(self.keypointmapper.keypoints) > 0:
            self.opengl_renderer.render_points(
                name="keypoints",
                points=self.keypoints_pos.numpy(),
                radius=1.,
                colors=(0.0, 1.0, 0.0),
                as_spheres=True,
                visible=True,
            )

        if self.trajctory_type == "kps":
            if self.step < len(self.target_trajectory):
                self.opengl_renderer.render_points(
                    name="target_keypoints",
                    points=self.target_trajectory[self.step].numpy(),
                    radius=1.,
                    colors=(1.0, 0.0, 0.0),
                    as_spheres=True,
                    visible=True,
                )
            else:
                self.opengl_renderer.render_points(
                    name="target_keypoints",
                    points=self.target_trajectory[-1].numpy(),
                    radius=1.,
                    colors=(1.0, 0.0, 0.0),
                    as_spheres=True,
                    visible=True,
                )

        self.opengl_renderer.end_frame()
        
        
        self.frame_id += 1
        
    def run_opengl_viewer(self, total_steps: int = None, total_time: int=None, fps: int = None, save_frames: bool = False, gradient_mode: bool = False):
        if not gradient_mode and not save_frames:
            interactive_mode = True
        else:
            interactive_mode = False
        self.gradient_mode = gradient_mode

        self.init_opengl_renderer(fps=fps, interactive_flag=interactive_mode)
        self.reset_to_rest()
        self.frame_id = 0

        self.step = 0
        self.total_time = total_time
        self.total_steps = total_steps

        while self.opengl_renderer.is_running():
            self.aspect = self.opengl_renderer.window.width / self.opengl_renderer.window.height
            self._handle_keyboard()
            if gradient_mode:
                loss, grad_E_log, grad_F, grad_F_amp, grad_damping_amp = self.run_with_tape(step = self.step)
                self.loss_hist.append(loss)
                self.grad_E_log_hist.append(grad_E_log)
                self.grad_F_hist.append(grad_F)
                self.grad_F_amp_hist.append(grad_F_amp)
                self.grad_damping_amp_hist.append(grad_damping_amp)
            else:
                loss = self.one_forward_step(step = self.step)
                self.loss_hist.append(loss)
                if save_frames:
                    self.x_hist.append(self.x_state_curr.numpy())
                    self.kps_hist.append(self.keypoints_pos.numpy())
            self.applied_force_np = self.applied_force_field.numpy()[0] * max(len(self.applied_idx), 1) / self.force_modulation

            self.render_opengl_frame(self.step)

            self.step += 1
            if self.total_steps is not None and self.step >= self.total_steps:
                if save_frames:
                    np.savez(f"data/Youngs_{int(self.youngs_np)}_time_{int(total_time)}_dt_{self.dt}_sweep_{self.sweep_count}_damping_amp_{self.damping_amp_np:.2e}.npz", 
                             position=np.array(self.x_hist), 
                             keypoints=np.array(self.kps_hist),
                             youngs = self.youngs_np,
                             youngs_log = self.youngs_log_np,
                             applied_force=self.per_vertex_force)
                break
        
        # self.opengl_renderer.close()

    def quat_from_arrow_to_force(self, dir_vec):
        if np.linalg.norm(dir_vec) < 1e-8:
            return (0.0, 0.0, 0.0, 1.0)

        dir_vec = np.asarray(dir_vec, dtype=np.float32)

        dir_vec /= np.linalg.norm(dir_vec)

        cross = np.cross(self.arrow_dir, dir_vec)

        dot = np.dot(self.arrow_dir, dir_vec)

        q = np.array([
            cross[0],
            cross[1],
            cross[2],
            1.0 + dot
        ])

        q /= np.linalg.norm(q)

        return tuple(q.astype(np.float32))

    def update_render_force_arrows(self,):
        if self.show_force_arrow and self.show_applied_points is not None and len(self.applied_idx_np) > 0:
            scales = np.linalg.norm(self.applied_force_field.numpy()[0])/(500 * self.series_force_amp_np)
            scales_tuple = tuple([scales for _ in range(3)])
            self.force_instancer.allocate_instances(
                positions=[tuple(self.x_state_curr.numpy()[vid]) for vid in self.applied_idx_np],
                rotations=[self.quat_from_arrow_to_force(self.applied_force_field.numpy()[0]) for _ in range(len(self.applied_idx_np))],
                colors1=[(1.0, 0.0, 0.0) for _ in range(len(self.applied_idx_np))],
                colors2=[(1.0, 0.0, 0.0) for _ in range(len(self.applied_idx_np))],
                scalings=[scales_tuple for _ in range(len(self.applied_idx_np))],
            )

    ### Update the camera position
    def _handle_keyboard(self,):
        key = pyglet.window.key
        if self.key_handler[key.LEFT]:
            self._update_orbit_camera_pitch(-1.)

        if self.key_handler[key.RIGHT]:
            self._update_orbit_camera_pitch(1.)

        if self.key_handler[key.UP]:
            self._update_orbit_camera_roll(1.)

        if self.key_handler[key.DOWN]:
            self._update_orbit_camera_roll(-1.)

        if self.key_handler[key.R]:
            self._reset_camera()

        if self.key_handler[key.SPACE]:
            self.opengl_renderer.paused = not self.opengl_renderer.paused

        if self.key_handler[key.ESCAPE]:
            self.opengl_renderer.close()

        if self.key_handler[key.I]:
            self.opengl_renderer.show_info = not self.opengl_renderer.show_info

        if self.key_handler[key.A]:
            self.opengl_renderer.draw_axis = not self.opengl_renderer.draw_axis

        if self.key_handler[key.S]:
            self.opengl_renderer.render_wireframe = not self.opengl_renderer.render_wireframe

    def _update_orbit_camera_roll(self, direction:float):
        off_set = self.camera_pos - self.camera_look_at

        c = np.cos(direction * self.rotational_angle)
        s = np.sin(direction * self.rotational_angle)

        y = off_set[1]
        z = off_set[2]

        off_set[1] = c * y - s * z
        off_set[2] = s * y + c * z

        self.camera_pos = self.camera_look_at + off_set

        self.camera_front = self.camera_look_at - self.camera_pos
        norm = np.linalg.norm(self.camera_front)
        if norm < 1e-12:
            self.camera_front = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            self.camera_front = self.camera_front / norm

        right = np.cross(self.camera_front, self.camera_up)
        self.camera_up = np.cross(right, self.camera_front)
        norm = np.linalg.norm(self.camera_up)
        self.camera_up = self.camera_up / norm
        # print(f"Camera Pos: {camera_pos}, Front: {camera_front}, Up: {camera_up}")
 
        self.opengl_renderer._camera_pos = PyVec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        self.opengl_renderer._camera_front = PyVec3(self.camera_front[0], self.camera_front[1], self.camera_front[2])
        self.opengl_renderer._camera_up = PyVec3(self.camera_up[0], self.camera_up[1], self.camera_up[2])

        self.opengl_renderer.update_view_matrix()

    def _update_orbit_camera_pitch(self, direction:float):
        off_set = self.camera_pos - self.camera_look_at

        c = np.cos(direction * self.rotational_angle)
        s = np.sin(direction * self.rotational_angle)

        x = off_set[0]
        y = off_set[1]
        
        off_set[0] = c * x - s * y
        off_set[1] = s * x + c * y

        self.camera_pos = self.camera_look_at + off_set

        self.camera_front = self.camera_look_at - self.camera_pos
        norm = np.linalg.norm(self.camera_front)
        if norm < 1e-12:
            self._reset_camera()
            return
        else:
            self.camera_front = self.camera_front / norm
            right = np.cross(self.camera_front, self.camera_up)
            self.camera_up = np.cross(right, self.camera_front)
            norm = np.linalg.norm(self.camera_up)
            if norm < 1e-12:
                self._reset_camera()
                return
            else:
                self.camera_up = self.camera_up / norm
 
        self.opengl_renderer._camera_pos = PyVec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        self.opengl_renderer._camera_front = PyVec3(self.camera_front[0], self.camera_front[1], self.camera_front[2])
        self.opengl_renderer._camera_up = PyVec3(self.camera_up[0], self.camera_up[1], self.camera_up[2])

        self.opengl_renderer.update_view_matrix()

    def _reset_camera(self):
        self.opengl_renderer._camera_pos = PyVec3(self.init_camera_pos[0], self.init_camera_pos[1], self.init_camera_pos[2])
        self.opengl_renderer._camera_front = PyVec3(self.init_camera_front[0], self.init_camera_front[1], self.init_camera_front[2])
        self.opengl_renderer._camera_up = PyVec3(self.init_camera_up[0], self.init_camera_up[1], self.init_camera_up[2])
        
        self.camera_pos = self.init_camera_pos.copy()
        self.camera_front = self.init_camera_front.copy()
        self.camera_up = self.init_camera_up.copy()

        self.opengl_renderer.update_view_matrix()

    def _sign_justification(self, value):
        if value == 0:
            return 'Zero'
        elif value > 0:
            return 'Positive'
        else:
            return 'Negative'

    def _draw_my_overlay(self,):
        self.my_label.text = f"Youngs: {self.youngs_np:.2f}"
        self.my_label.text += f"\nYoungs Log: {self.youngs_log_np:.2f}"
        self.my_label.text += f"\nStep: {self.step+1}/{self.total_steps}\nTimes: {(self.step+1)*self.dt:.2f}/{self.total_time}s"
        self.my_label.text += f"\nApplied Force: ({self.applied_force_np[0]:.2e}, {self.applied_force_np[1]:.2e}, {self.applied_force_np[2]:.2e})"
        self.my_label.text += f"\nF_free_elastic: ({self.total_free_node_force[0]:.2e}, {self.total_free_node_force[1]:.2e}, {self.total_free_node_force[2]:.2e})"
        self.my_label.text += f"\nF_fix_elastic: ({self.total_fix_node_force[0]:.2e}, {self.total_fix_node_force[1]:.2e}, {self.total_fix_node_force[2]:.2e})"
        self.my_label.text += f"\nF_total_elastic: ({self.total_node_force[0]:.2e}, {self.total_node_force[1]:.2e}, {self.total_node_force[2]:.2e})"
        self.my_label.text += f"\nF_applied: ({self.applied_node_force[0]:.2e}, {self.applied_node_force[1]:.2e}, {self.applied_node_force[2]:.2e})"
        if self.target_trajectory is not None:
                        self.my_label.text += f"\n\nLoss ({self._sign_justification(np.sum(self.loss_hist))}):\nCurrent: {self.loss_hist[-1]:.2e},\nAverage: {np.mean(self.loss_hist):.2e}\nTotal: {np.sum(self.loss_hist):.2e}\n"
        if self.gradient_mode:
            total_grad_F = np.array(self.grad_F_hist).reshape(-1, 3).sum(axis=0)
            self.my_label.text += f"\nYoungs Modulus Log Gradient ({self._sign_justification(np.sum(self.grad_E_log_hist))}):\nCurrent: {self.grad_E_log_hist[-1]:.2e},\nTotal: {np.sum(self.grad_E_log_hist):.2e}\n"
            self.my_label.text += f"\nForce Amplification Gradient ({self._sign_justification(np.sum(self.grad_F_amp_hist))}):\nCurrent: {self.grad_F_amp_hist[-1]:.2e},\nTotal: {np.sum(self.grad_F_amp_hist):.2e}\n"
            self.my_label.text += f"\nDamping Amplification Gradient ({self._sign_justification(np.sum(self.grad_damping_amp_hist))}):\nCurrent: {self.grad_damping_amp_hist[-1]:.2e},\nTotal: {np.sum(self.grad_damping_amp_hist):.2e}"
            # self.my_label.text += f"\nCurrent Grad F: ({self.grad_F_hist[-1][0]:.2e}, {self.grad_F_hist[-1][1]:.2e}, {self.grad_F_hist[-1][2]:.2e})\nTotal Grad F: ({total_grad_F[0]:.2e}, {total_grad_F[1]:.2e}, {total_grad_F[2]:.2e})"
        self.my_label.x = 10
        self.my_label.y = self.opengl_renderer._info_label.y - self.opengl_renderer._info_label.content_height - 5
        self.my_label.draw()

    def _my_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos[:] = [x, y]

    def _my_mouse_press(self, x, y, button, modifiers):
        self.mouse_pos[:] = [x, y]
        self.ray_o, self.ray_d = self._screen_to_world_ray(self.mouse_pos, self.camera_pos, self.camera_look_at,
                                                 self.camera_up, self.aspect, self.opengl_renderer.camera_fov)
        self.selected_vid = self._pick_vertex_from_ray(self.ray_o, self.ray_d)
        if self.selected_vid >=0:
            x_curr = self.x_state_curr.numpy()[self.selected_vid].astype(np.float32)
            self.drag_plane_point = x_curr.copy()
            forward = self._normalize_np(self.camera_look_at - self.camera_pos)
            self.drag_plane_normal = forward.copy()

    def _my_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse_pos[:] = [x, y]
        if self.selected_vid >= 0:
            ray_o, ray_d = self._screen_to_world_ray(
                self.mouse_pos, self.camera_pos, self.camera_look_at, self.camera_up, self.aspect, self.opengl_renderer.camera_fov)
            hit = self._intersect_ray_plane(
                ray_o, ray_d,
                self.drag_plane_point,
                self.drag_plane_normal)
            if hit is not None:
                wp.launch(set_single_vertex_position,
                        dim=1,
                        inputs=[self.x_state_curr, int(self.selected_vid), float(hit[0]), float(hit[1]), float(hit[2])],
                        device=self.device,)

    def _my_mouse_release(self, x, y, button, modifiers):
        self.mouse_pos[:] = [x, y]
        self.drag_plane_point = None
        self.drag_plane_normal = None
        self.selected_vid = -1

    def _my_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera_pos += scroll_y * self.camera_front
        self.opengl_renderer._camera_pos = PyVec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        self.opengl_renderer._camera_front = PyVec3(self.camera_front[0], self.camera_front[1], self.camera_front[2])
        self.opengl_renderer._camera_up = PyVec3(self.camera_up[0], self.camera_up[1], self.camera_up[2])

        self.opengl_renderer.update_view_matrix()

    # =========================================================
    # OpenGL GUI: dragging helpers
    # =========================================================
    def _normalize_np(self, v):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return v.copy()
        return v / n

    def _camera_basis(self, cam_pos, cam_lookat, cam_up):
        forward = self._normalize_np(cam_lookat - cam_pos)
        right = self._normalize_np(np.cross(forward, cam_up))
        true_up = self._normalize_np(np.cross(right, forward))
        return forward, right, true_up

    def _screen_to_world_ray(self, mouse_pos, cam_pos, cam_lookat, cam_up, aspect, fov_y_deg=45.0):
        """
        mouse_pos: (u, v) in [0,1]x[0,1]
        returns:
            ray_origin, ray_dir
        """
        u, v = mouse_pos
        ndc_x = (2.0 * (u/self.opengl_renderer.window.width) - 1.0)
        ndc_y = (2.0 * (v/self.opengl_renderer.window.height) - 1.0)

        forward, right, true_up = self._camera_basis(cam_pos, cam_lookat, cam_up)

        tan_half_fov = np.tan(np.deg2rad(fov_y_deg) * 0.5)
        px = ndc_x * tan_half_fov * aspect
        py = ndc_y * tan_half_fov

        ray_dir = self._normalize_np(forward + px * right + py * true_up)
        ray_origin = cam_pos.copy()
        return ray_origin, ray_dir

    def _pick_vertex_from_ray(self, ray_origin, ray_dir, pick_radius=None):
        """
        Find nearest draggable vertex to the ray.
        """
        if pick_radius is None:
            pick_radius = self.drag_pick_radius

        best_vid = -1
        best_dist2 = float("inf")
        r2 = pick_radius * pick_radius

        self.x_np = self.x_state_curr.numpy()
        for vid in range(self.mesh.nv):
            if self.non_draggable_mask_np[vid]:
                continue

            p = self.x_np[vid]
            w = p - ray_origin
            t = np.dot(w, ray_dir)
            if t < 0.0:
                continue

            closest = ray_origin + t * ray_dir
            d2 = np.sum((p - closest) ** 2)

            if d2 < r2 and d2 < best_dist2:
                best_dist2 = d2
                best_vid = vid

        return best_vid

    def _intersect_ray_plane(self, ray_origin, ray_dir, plane_point, plane_normal):
        denom = float(np.dot(ray_dir, plane_normal))
        if abs(denom) < 1e-10:
            return None

        t = float(np.dot(plane_point - ray_origin, plane_normal) / denom)
        if t < 0.0:
            return None

        return ray_origin + t * ray_dir

    # -----------------------------------------------------
    # USD export / visualization
    # -----------------------------------------------------
    def init_usd_renderer(self, usd_path: str, fps: float = 60.0, scaling: float = 1.0):
        """
        Create a Warp USD renderer and register the deformable boundary mesh.

        Notes:
        - The rendered mesh uses boundary triangles from MeshLoader3D_Warp.
        - Vertex positions are updated every frame with update_shape_vertices().
        - Call save_usd_frame() after each simulation step you want to record.
        - Call finalize_usd() once at the end.
        """
        if self.mesh.boundary_face_idx_np is None:
            raise RuntimeError("Boundary faces are not available. Make sure build_shared_faces() has been run in the mesh loader.")

        self.usd_path = usd_path
        self.usd_scale = float(scaling)
        os.makedirs(os.path.dirname(os.path.abspath(usd_path)), exist_ok=True)

        self.usd_renderer = warp.render.UsdRenderer(
            usd_path,
            up_axis="x",
            fps=float(fps),
            scaling=float(scaling),
        )

        points = self.x_state_curr.numpy()
        self.usd_renderer.begin_frame(0.0)
        self.usd_renderer.render_mesh(
            name=self.usd_mesh_name,
            colors=(0.6, 0.8, 1.0),
            points=points,
            indices=self.boundary_indices.reshape(-1),
            update_topology=True,)

        self.usd_renderer.end_frame()
        self.usd_renderer.save()

    def save_usd_frame(self, frame_index: int, x_override=None):
        """
        Write one animation frame to the USD file.

        Parameters
        ----------
        frame_index : int
            Frame number in the exported USD animation.
        x_override : wp.array(dtype=wp.vec3) or None
            If provided, export this state instead of self.x_state.
        """
        if self.usd_renderer is None:
            raise RuntimeError("USD renderer is not initialized. Call init_usd_renderer() first.")

        points = self.x_state_curr.numpy() if x_override is None else x_override

        time_in_seconds = float(frame_index) / float(self.usd_renderer.fps)
        self.usd_renderer.begin_frame(time_in_seconds)
        self.usd_renderer.render_mesh(
            name=self.usd_mesh_name,
            colors=(0.6, 0.8, 1.0),
            points=points,
            indices=self.boundary_indices.reshape(-1),
            update_topology=(frame_index == 0),
        )
        self.usd_renderer.end_frame()

    def export_usd_sequence(self, num_frames: int, usd_path: str, fps: float = 60.0):
        """
        Run forward simulation and export the deformed boundary mesh to a USD file.
        This is useful for visually checking whether the simulation stays stable.
        """
        self.init_usd_renderer(usd_path=usd_path, fps=fps)
        self.reset_to_rest()
        self.save_usd_frame(0)

        for f in range(1, num_frames + 1):
            self.one_forward_step()
            self.save_usd_frame(f)

        self.finalize_usd()
        return self.usd_path

    def finalize_usd(self):
        if self.usd_renderer is not None:
            self.usd_renderer.save()

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    def run_one_XPBD_loop(self, 
                          total_steps: int=100):
        for step in range(total_steps):
            loss, grad_E_log, grad_F, grad_F_amp, grad_damping_amp = self.run_with_tape(step = step)
            self.loss_hist.append(loss)
            self.grad_E_log_hist.append(grad_E_log)
            self.grad_F_hist.append(grad_F)
            self.grad_F_amp_hist.append(grad_F_amp)
            self.grad_damping_amp_hist.append(grad_damping_amp)

        if len(self.grad_damping_amp_hist) > total_steps:
            raise ValueError("Storage lists for the loss and gradients are not initialized properly.")
        
        avg_loss = np.mean(self.loss_hist)
        total_loss = np.sum(self.loss_hist)
        total_grad_E_log = np.sum(self.grad_E_log_hist)
        total_grad_F = np.array(self.grad_F_hist).reshape(-1, 3).sum(axis=0)
        total_grad_F_amp = np.sum(self.grad_F_amp_hist)
        total_grad_damping_amp = np.sum(self.grad_damping_amp_hist)
        return avg_loss, total_loss, total_grad_E_log, total_grad_F, total_grad_F_amp, total_grad_damping_amp

    def train(self,
              project_name: str = "default_project",
              contact_pos: str = "mid",
              stop_condition: str = "thresholding",
              convergence_patience: int = 5,
              relative_change_threshold: float = 1.e-4,
              optimized_method: str = None,
              alternating_epochs: int = 20,
              max_epochs: int = 20, 
              lr: List[float] = [1e-3, 1e-3], 
              eps: float = 1.0, 
              optimize_subject: List[str] = ["Youngs_Modulus", "Damping_Amplification"],
              total_steps: int = 100, 
              optimizer: str = None):
        
        avg_loss_history = []
        patience = convergence_patience
        converged_count = 0
        relative_change = 0.
        maximum_epoch_flag = False

        self.optimizer = None
        if optimizer == "Adam":
            self.optimizer = [AdamOptimizer(lr=lr[i]) for i in range(len(optimize_subject))]

        train_root_dir = f"data/training_logs/{contact_pos.capitalize()}/{project_name}"
        gif_save_path = f"{train_root_dir}/gifs"
        train_info_path = f"{train_root_dir}/train_info.json"
        log_path = f"{train_root_dir}/log.json"

        if os.path.exists(train_root_dir):
            shutil.rmtree(train_root_dir)

        os.makedirs(train_root_dir, exist_ok=True)
        os.makedirs(gif_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(train_info_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        train_info = {}
        train_info["total_time"] = total_steps * self.dt
        train_info["applied_center_height"] = self.applied_center_height
        train_info["optimize_subject"] = optimize_subject
        train_info["lr"] = lr
        train_info["optimized_method"] = optimized_method
        if optimized_method == "alternating":
            train_info["alternating_epochs"] = alternating_epochs
        train_info["max_epochs"] = max_epochs
        train_info["stop_condition"] = stop_condition
        if stop_condition == "convergence" or stop_condition == "both":
            train_info["convergence_patience"] = convergence_patience
            train_info["relative_change_threshold"] = relative_change_threshold
        train_info["avg_loss_eps"] = eps

        with open(train_info_path, 'w') as f:
            json.dump(train_info, f, indent=4, sort_keys=False)

        opt_subjects_num = max(len(optimize_subject), 1)

        if opt_subjects_num < 2 and optimized_method == "alternating":
            raise ValueError("Alternating optimization requires at least 2 subjects to optimize.")

        log_data = {}
        for subject in optimize_subject:
            log_data[subject] = []
            if subject == "Youngs_Modulus":
                log_data[subject].append(self.youngs_np)
                log_data["Youngs_Modulus_log"] = []
                log_data["Youngs_Modulus_log"].append(self.youngs_log_np)
            elif subject == "Applied_Force":
                log_data[subject].append(self.per_vertex_force)
            elif subject == "Force_Amplification":
                log_data[subject].append(self.series_force_amp_np)
            elif subject == "Damping_Amplification":
                log_data[subject].append(self.damping_amp_np)

        log_data["avg_loss"] = []
        log_data["total_loss"] = []
        log_data["relative_avg_loss_change"] = []
        avg_loss = 1e3
        epoch = 0
        while epoch <= max_epochs:
            self.reset_to_rest()
            avg_loss, total_loss, total_grad_E_log, total_grad_F, total_grad_F_amp, total_grad_damping_amp = self.run_one_XPBD_loop(total_steps=total_steps)

            avg_loss_history.append(avg_loss)
            log_data["avg_loss"].append(avg_loss)
            log_data["total_loss"].append(total_loss)

            # Convergence check
            if len(avg_loss_history) > 1:
                relative_change = abs(avg_loss_history[-1] - avg_loss_history[-2]) / max(abs(avg_loss_history[-2]), 1e-8)
                if relative_change < relative_change_threshold:
                    converged_count += 1
                else:
                    converged_count = 0

            log_data["relative_avg_loss_change"].append(relative_change)

            if epoch == max_epochs:
                maximum_epoch_flag = True
                print("Final optimized parameters performance:")
            else:
                print(f"Epoch {epoch}:")
            print(f"Current Young's Modulus Log: {self.youngs_log_np:.2f}, Current Young's Modulus: {self.youngs_np:.2f}, Current Damping Amplification: {self.damping_amp_np:.2f}, Current Force Amplification: {self.series_force_amp_np:.2f}")
            print(f"Average_loss={avg_loss:.4e}, relative_avg_loss_change={relative_change * 100:.2f}%, total_loss={total_loss:.4e}, total_grad_E_log={total_grad_E_log:.4e}, total_grad_F_amp={total_grad_F_amp:.4e}, total_grad_damping_amp={total_grad_damping_amp:.4e}")

            if stop_condition == "convergence":
                if converged_count >= patience:
                    print(f"Convergence achieved at epoch {epoch} with average loss {avg_loss:.2e}. Stopping training.")
                    break
            elif stop_condition == "thresholding":
                if avg_loss < eps:
                    print(f"Loss threshold achieved at epoch {epoch} with average loss {avg_loss:.2e}. Stopping training.")
                    break
            elif stop_condition == "both":
                if converged_count >= patience and avg_loss < eps:
                    print(f"Both convergence and loss threshold achieved at epoch {epoch}. Stopping training.")
                    break

            if maximum_epoch_flag:
                print(f"Reached maximum epoch: {max_epochs}. Stopping training.")
                break

            for i, subject in enumerate(optimize_subject):
                if optimized_method == "alternating":
                    if i != (epoch // alternating_epochs) % opt_subjects_num:
                        optimize_flag = False
                    else:
                        optimize_flag = True
                elif optimized_method == None or optimized_method == "simultaneous":
                    optimize_flag = True
                if subject == "Youngs_Modulus":
                    if optimize_flag:
                        if self.optimizer is not None:
                            self.youngs_log_np = self.optimizer[i].step(self.youngs_log_np, total_grad_E_log)
                        else:
                            self.youngs_log_np -= lr[i] * total_grad_E_log
                        print(f"Updating {subject}...")
                    self.youngs_np = np.exp(self.youngs_log_np)

                    log_data["Youngs_Modulus_log"].append(self.youngs_log_np)
                    log_data[subject].append(self.youngs_np)

                elif subject == "Applied_Force":
                    if optimize_flag:
                        if self.optimizer is not None:
                            self.per_vertex_force = self.optimizer[i].step(self.per_vertex_force, total_grad_F)
                        else:
                            self.per_vertex_force -= lr[i] * total_grad_F
                        print(f"Updating {subject}...") 

                    log_data[subject].append(self.per_vertex_force)

                elif subject == "Force_Amplification":
                    if optimize_flag:
                        if self.optimizer is not None:
                            self.series_force_amp_np = self.optimizer[i].step(self.series_force_amp_np, total_grad_F_amp)
                        else:
                            self.series_force_amp_np -= lr[i] * total_grad_F_amp
                        print(f"Updating {subject}...")
                        
                    log_data[subject].append(self.series_force_amp_np)

                elif subject == "Damping_Amplification": ## Damping Amplification is clamped to be non-negative
                    if optimize_flag:
                        if self.optimizer is not None:
                            self.damping_amp_np = self.optimizer[i].step(self.damping_amp_np, total_grad_damping_amp)
                        else:
                            self.damping_amp_np -= lr[i] * total_grad_damping_amp
                        if self.damping_amp_np < 0:
                            self.damping_amp_np = 1.e-2
                        print(f"Updating {subject}...")
                    log_data[subject].append(self.damping_amp_np)
            print()
            epoch += 1

       # Save the training log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4, sort_keys=False)

        print(f"Final Young's Modulus Log: {self.youngs_log_np:.2f}, Final Young's Modulus: {self.youngs_np:.2f}, Final Damping Amplification: {self.damping_amp_np:.2f}, Final Force Amplification: ({self.series_force_amp_np:.2f})")
        return