import numpy as np
import warp as wp
from utils.MeshLoader import MeshLoader3D_Warp

# =========================================================
# kernels
# =========================================================

@wp.kernel
def compute_tet_volume_kernel(
    topo: wp.array(dtype=wp.vec4i),
    X: wp.array(dtype=wp.vec3),
    tet_volume: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    tet = topo[t]

    x1 = X[tet[0]]
    x2 = X[tet[1]]
    x3 = X[tet[2]]
    x4 = X[tet[3]]

    Dm = wp.matrix_from_cols(x1 - x4, x2 - x4, x3 - x4)
    tet_volume[t] = wp.abs(wp.determinant(Dm)) / 6.0


@wp.kernel
def precompute_tet_Dm_kernel(
    topo: wp.array(dtype=wp.vec4i),
    X: wp.array(dtype=wp.vec3),
    Dm: wp.array(dtype=wp.mat33),
    Dm_inv: wp.array(dtype=wp.mat33),
    det_Dm: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    tet = topo[t]

    x1 = X[tet[0]]
    x2 = X[tet[1]]
    x3 = X[tet[2]]
    x4 = X[tet[3]]

    dm = wp.matrix_from_cols(x1 - x4, x2 - x4, x3 - x4)
    Dm[t] = dm
    det_Dm[t] = wp.determinant(dm)
    Dm_inv[t] = wp.inverse(dm)


@wp.kernel
def clear_int_array_kernel(arr: wp.array(dtype=wp.int32)):
    i = wp.tid()
    arr[i] = 0


@wp.kernel
def compute_vertex_count_kernel(
    topo: wp.array(dtype=wp.vec4i),
    vertex_count: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    tet = topo[t]

    wp.atomic_add(vertex_count, tet[0], 1)
    wp.atomic_add(vertex_count, tet[1], 1)
    wp.atomic_add(vertex_count, tet[2], 1)
    wp.atomic_add(vertex_count, tet[3], 1)


@wp.kernel
def compute_edge_len_kernel(
    E: wp.array(dtype=wp.vec2i),
    X: wp.array(dtype=wp.vec3),
    edge_len: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    ij = E[e]

    x1 = X[ij[0]]
    x2 = X[ij[1]]
    edge_len[e] = wp.length(x1 - x2)


@wp.kernel
def points_in_aabb_mask_kernel(
    X: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.int32),
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    eps: float,
):
    i = wp.tid()
    p = X[i]

    lx = p[0] - cx
    ly = p[1] - cy
    lz = p[2] - cz

    inside = (
        wp.abs(lx) <= hx + eps
        and wp.abs(ly) <= hy + eps
        and wp.abs(lz) <= hz + eps
    )

    mask[i] = 1 if inside else 0


@wp.kernel
def points_in_obb_mask_kernel(
    X: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.int32),
    cx: float,
    cy: float,
    cz: float,
    r00: float, r01: float, r02: float,
    r10: float, r11: float, r12: float,
    r20: float, r21: float, r22: float,
    hx: float,
    hy: float,
    hz: float,
    eps: float,
):
    i = wp.tid()
    p = X[i]

    R = wp.mat33(
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    )

    d = wp.vec3(p[0] - cx, p[1] - cy, p[2] - cz)

    # world -> local
    q = wp.transpose(R) * d

    inside = (
        wp.abs(q[0]) <= hx + eps
        and wp.abs(q[1]) <= hy + eps
        and wp.abs(q[2]) <= hz + eps
    )

    mask[i] = 1 if inside else 0


# =========================================================
# host-side class
# =========================================================

class PropertyCalculator3D_Warp:
    def __init__(
        self,
        mesh_loader: MeshLoader3D_Warp,
        device=None,
        compliance_modulation=1e-2,
        float_dtype=wp.float32,
        int_dtype=wp.int32,
    ):
        """
        mesh_loader: MeshLoader3D_Warp
            mesh_loader.X    : wp.array(dtype=wp.vec3), shape (nv,)
            mesh_loader.Topo : wp.array(dtype=wp.vec4i), shape (nt,)
            mesh_loader.E    : wp.array(dtype=wp.vec2i), shape (ne,)
        """
        self.m = mesh_loader
        self.device = device if device is not None else self.m.device
        self.compliance_modulation = compliance_modulation
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        if self.float_dtype == wp.float64:
            self.dtype_np = np.float64
        else:
            self.dtype_np = np.float32

        # outputs
        self.tet_volume = None      # wp.array(dtype=float)
        self.edge_len = None        # wp.array(dtype=float)
        self.obb_mask = None        # wp.array(dtype=int32)
        self.aabb_mask = None       # wp.array(dtype=int32)

        self.Dm = None              # wp.array(dtype=wp.mat33)
        self.Dm_inv = None          # wp.array(dtype=wp.mat33)
        self.det_Dm = None          # wp.array(dtype=float)

        self.vertex_count = None    # wp.array(dtype=int32)
        self.avg_len = None         # python float

        self.pre_cal_tet_volume()
        self.precompute_tet_Dm()
        self.compute_vertex_count()
        self.average_edge()

    # =========================================================
    # 1) tetra volume
    # =========================================================
    def alloc_tet_volume(self):
        self.tet_volume = wp.zeros(
            shape=(self.m.nt,),
            dtype=self.float_dtype,
            device=self.device,
        )

    def pre_cal_tet_volume(self):
        if self.tet_volume is None:
            self.alloc_tet_volume()

        wp.launch(
            kernel=compute_tet_volume_kernel,
            dim=self.m.nt,
            inputs=[self.m.Topo, self.m.X, self.tet_volume],
            device=self.device,
        )
        return self.tet_volume

    # =========================================================
    # 2) Dm, Dm_inv, det_Dm
    # =========================================================
    def alloc_tet_Dm(self):
        self.Dm = wp.zeros(
            shape=(self.m.nt,),
            dtype=wp.mat33,
            device=self.device,
        )
        self.Dm_inv = wp.zeros(
            shape=(self.m.nt,),
            dtype=wp.mat33,
            device=self.device,
        )
        self.det_Dm = wp.zeros(
            shape=(self.m.nt,),
            dtype=self.float_dtype,
            device=self.device,
        )

    def precompute_tet_Dm(self):
        if self.Dm is None or self.Dm_inv is None or self.det_Dm is None:
            self.alloc_tet_Dm()

        wp.launch(
            kernel=precompute_tet_Dm_kernel,
            dim=self.m.nt,
            inputs=[self.m.Topo, self.m.X, self.Dm, self.Dm_inv, self.det_Dm],
            device=self.device,
        )
        return self.Dm, self.Dm_inv, self.det_Dm

    # =========================================================
    # 3) vertex count
    # =========================================================
    def alloc_vertex_count(self):
        self.vertex_count = wp.zeros(
            shape=(self.m.nv,),
            dtype=wp.int32,
            device=self.device,
        )

    def compute_vertex_count(self):
        if self.vertex_count is None:
            self.alloc_vertex_count()

        wp.launch(
            kernel=clear_int_array_kernel,
            dim=self.m.nv,
            inputs=[self.vertex_count],
            device=self.device,
        )

        wp.launch(
            kernel=compute_vertex_count_kernel,
            dim=self.m.nt,
            inputs=[self.m.Topo, self.vertex_count],
            device=self.device,
        )

        return self.vertex_count

    # =========================================================
    # 4) edge length / average edge length
    # =========================================================
    def alloc_edge_len(self):
        if self.m.E is None:
            raise RuntimeError("mesh_loader.E is None. Call mesh_loader.build_edges() first.")

        self.edge_len = wp.zeros(
            shape=(self.m.ne,),
            dtype=self.float_dtype,
            device=self.device,
        )
        self.avg_len = 0.0

    def compute_edge_len(self):
        if self.edge_len is None:
            self.alloc_edge_len()

        wp.launch(
            kernel=compute_edge_len_kernel,
            dim=self.m.ne,
            inputs=[self.m.E, self.m.X, self.edge_len],
            device=self.device,
        )
        return self.edge_len

    def average_edge(self) -> float:
        if self.edge_len is None:
            self.alloc_edge_len()

        self.compute_edge_len()
        edge_len_np = self.edge_len.numpy()
        self.avg_len = float(edge_len_np.mean())
        return self.avg_len

    # =========================================================
    # 5) masks
    # =========================================================
    def alloc_masks(self):
        self.obb_mask = wp.zeros(
            shape=(self.m.nv,),
            dtype=wp.int32,
            device=self.device,
        )
        self.aabb_mask = wp.zeros(
            shape=(self.m.nv,),
            dtype=wp.int32,
            device=self.device,
        )

    # =========================================================
    # 6) AABB in 3D
    # =========================================================
    def points_in_aabb_3d_batch(self, center, half_extents_list, eps=1e-6, restrict_contact_surf=True):
        mask = np.zeros(self.m.nv, dtype=bool)
        for i in range(len(center)):
            mask_np, __ = self.points_in_aabb_3d(center[i], half_extents_list[i], eps, restrict_contact_surf=restrict_contact_surf)
            mask |= mask_np
        idx = np.where(mask)[0]
        return mask, idx


    def points_in_aabb_3d(self, center, half_extents, eps=1e-6, restrict_contact_surf=True):
        if self.aabb_mask is None:
            self.alloc_masks()

        cx, cy, cz = map(float, center)
        hx, hy, hz = map(float, half_extents)

        wp.launch(
            kernel=points_in_aabb_mask_kernel,
            dim=self.m.nv,
            inputs=[
                self.m.X,
                self.aabb_mask,
                cx, cy, cz,
                hx, hy, hz,
                float(eps),
            ],
            device=self.device,
        )

        mask_np = self.aabb_mask.numpy().astype(bool).reshape(-1)
        idx_np = np.where(mask_np)[0]

        if restrict_contact_surf:
            idx_np = idx_np[np.isin(idx_np, self.m.surface_vertices)]

        idx_np = idx_np[np.abs(self.m.X_np[idx_np][:, 0]-18.85565662)<1.0e-2]
        mask_np = np.isin(np.arange(self.m.nv), idx_np)

        return mask_np, idx_np

    # =========================================================
    # 7) OBB in 3D
    # =========================================================
    def points_in_obb_3d_batch(self, center, rotation_lists, half_extents_list, eps=1e-6):
        mask = np.zeros(self.m.nv, dtype=bool)
        for i in range(len(center)):
            mask_np, __ = self.points_in_obb_3d(center[i], rotation_lists[i], half_extents_list[i], eps)
            mask |= mask_np
        idx = np.where(mask)[0]
        return mask, idx

    def points_in_obb_3d(self, center, rotation_list, half_extents, eps=1e-6):
        if self.obb_mask is None:
            self.alloc_masks()

        c = np.asarray(center, dtype=np.float32).reshape(3)
        R = self.rotate_x(rotation_list[0]) @ self.rotate_y(rotation_list[1]) @ self.rotate_z(rotation_list[2])
        h = np.asarray(half_extents, dtype=np.float32).reshape(3)

        wp.launch(
            kernel=points_in_obb_mask_kernel,
            dim=self.m.nv,
            inputs=[
                self.m.X,
                self.obb_mask,
                float(c[0]), float(c[1]), float(c[2]),
                float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
                float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
                float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
                float(h[0]), float(h[1]), float(h[2]),
                float(eps),
            ],
            device=self.device,
        )

        mask_np = self.obb_mask.numpy().astype(bool).reshape(-1)
        idx_np = np.where(mask_np)[0]
        return mask_np, idx_np

    # =========================================================
    # 8) Rotation utilities
    # =========================================================
    def rotate_x(self, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        c = np.cos(angle_radians)
        s = np.sin(angle_radians)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]], dtype=np.float32)
    
    def rotate_y(self, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        c = np.cos(angle_radians)
        s = np.sin(angle_radians)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)

    def rotate_z(self, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        c = np.cos(angle_radians)
        s = np.sin(angle_radians)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]], dtype=np.float32)
    
    # =========================================================
    # 9) convenience / debug
    # =========================================================
    def summary(self):
        print("===== PropertyCalculator3D_Warp =====")
        print(f"device             : {self.device}")
        print(f"compliance_mod     : {self.compliance_modulation}")
        print(f"tet_volume alloc    : {self.tet_volume is not None}")
        print(f"Dm alloc            : {self.Dm is not None}")
        print(f"Dm_inv alloc        : {self.Dm_inv is not None}")
        print(f"det_Dm alloc        : {self.det_Dm is not None}")
        print(f"vertex_count alloc  : {self.vertex_count is not None}")
        print(f"edge_len alloc      : {self.edge_len is not None}")
        print(f"aabb_mask alloc     : {self.aabb_mask is not None}")
        print(f"obb_mask alloc      : {self.obb_mask is not None}")
        print("=====================================")