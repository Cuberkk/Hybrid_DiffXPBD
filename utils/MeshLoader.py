import numpy as np
import meshio
import warp as wp


class MeshLoader3D_Warp:
    def __init__(self, device=None, float_dtype=wp.float32, int_dtype=wp.int32):
        self.device = device if device is not None else wp.get_preferred_device()
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        if self.float_dtype == wp.float64:
            self.dtype_np = np.float64
        else:
            self.dtype_np = np.float32

        # -----------------------------
        # Host-side numpy copies
        # -----------------------------
        self.X_np = None              # (nv, 3) float
        self.Topo_np = None           # (nt, 4) int32

        self.E_np = None              # (ne, 2) int32
        self.Et_np = None             # (nt, 6) int32

        self.F_np = None              # (nf, 3) int32
        self.Ft_np = None             # (nt, 4) int32

        self.shared_face_idx_np = None       # (nsf,)
        self.face_to_tets_np = None          # (nsf, 2)
        self.boundary_face_idx_np = None     # (nbf,)
        self.boundary_face_to_tet_np = None  # (nbf,)
        self.opposite_vertices_np = None     # (nsf, 2)

        # -----------------------------
        # Warp arrays
        # -----------------------------
        self.X = None                 # wp.array(dtype=wp.vec3)
        self.Topo = None              # wp.array(dtype=wp.vec4i)

        self.E = None                 # wp.array(dtype=wp.vec2i)
        self.Et = None                # wp.array(dtype=vec6i custom type)

        self.F = None                 # wp.array(dtype=wp.vec3i)
        self.Ft = None                # wp.array(dtype=wp.vec4i)

        self.shared_face_idx = None       # wp.array(dtype=wp.int32)
        self.face_to_tets = None          # wp.array(dtype=wp.vec2i)
        self.boundary_face_idx = None     # wp.array(dtype=wp.int32)
        self.boundary_face_to_tet = None  # wp.array(dtype=wp.int32)
        self.opposite_vertices = None     # wp.array(dtype=wp.vec2i)

        # mesh size
        self.nv = 0
        self.ne = 0
        self.nf = 0
        self.nt = 0

        # Warp does not have a built-in vec6i alias like vec3i/vec4i,
        # so we define one here.
        self.vec6i = wp.types.vector(length=6, dtype=wp.int32)

    # =========================================================
    # Internal helper: numpy -> warp
    # =========================================================
    def _to_wp_vec_array(self, arr_np: np.ndarray, dtype):
        return wp.array(arr_np, dtype=dtype, device=self.device)

    def _upload_mesh_arrays(self):
        if self.X_np is not None:
            self.X = self._to_wp_vec_array(self.X_np, wp.vec3)

        if self.Topo_np is not None:
            self.Topo = self._to_wp_vec_array(self.Topo_np, wp.vec4i)

    def _upload_edge_arrays(self):
        if self.E_np is not None:
            self.E = self._to_wp_vec_array(self.E_np, wp.vec2i)

        if self.Et_np is not None:
            self.Et = self._to_wp_vec_array(self.Et_np, self.vec6i)

    def _upload_face_arrays(self):
        if self.F_np is not None:
            self.F = self._to_wp_vec_array(self.F_np, wp.vec3i)

        if self.Ft_np is not None:
            self.Ft = self._to_wp_vec_array(self.Ft_np, wp.vec4i)

    def _upload_shared_face_arrays(self):
        if self.shared_face_idx_np is not None:
            self.shared_face_idx = wp.array(
                self.shared_face_idx_np, dtype=wp.int32, device=self.device
            )

        if self.face_to_tets_np is not None:
            self.face_to_tets = self._to_wp_vec_array(self.face_to_tets_np, wp.vec2i)

        if self.boundary_face_idx_np is not None:
            self.boundary_face_idx = wp.array(
                self.boundary_face_idx_np, dtype=wp.int32, device=self.device
            )

        if self.boundary_face_to_tet_np is not None:
            self.boundary_face_to_tet = wp.array(
                self.boundary_face_to_tet_np, dtype=wp.int32, device=self.device
            )

        if self.opposite_vertices_np is not None:
            self.opposite_vertices = self._to_wp_vec_array(self.opposite_vertices_np, wp.vec2i)

    # =========================================================
    # Load gmsh tetrahedral mesh
    # =========================================================
    def load_gmsh(self, filepath: str, target_unit: str = "mm"):
        """
        Load a tetrahedral volume mesh from .msh using meshio.

        After calling:
            self.X    : wp.array(dtype=wp.vec3), shape (nv,)
            self.Topo : wp.array(dtype=wp.vec4i), shape (nt,)
        """
        mesh = meshio.read(filepath)

        pts = mesh.points
        if pts.shape[1] < 3:
            raise ValueError("Mesh points must have 3 coordinates for a 3D mesh.")

        verts_np = pts[:, :3].astype(self.dtype_np)

        if target_unit == "m":
            verts_np *= 1e-3
        elif target_unit == "mm":
            pass
        else:
            raise ValueError("target_unit must be 'mm' or 'm'.")

        if "tetra" in mesh.cells_dict:
            topo_np = mesh.cells_dict["tetra"].astype(np.int32)
        else:
            raise ValueError("No tetra cells found in the mesh.")

        # Convert 1-based indexing to 0-based if necessary
        if topo_np.min() == 1:
            topo_np -= 1

        self.X_np = np.ascontiguousarray(verts_np)
        self.Topo_np = np.ascontiguousarray(topo_np)

        self.nv = self.X_np.shape[0]
        self.nt = self.Topo_np.shape[0]

        self._upload_mesh_arrays()

        self.build_edges()
        self.build_faces()
        self.build_shared_faces()
        self.get_surface_vertices_numpy()

    # =========================================================
    # Build unique edges
    # =========================================================
    def build_edges(self):
        """
        For tetrahedra:
            local edges = 6
                (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)

        Output:
            self.E  : (ne,) wp.vec2i
            self.Et : (nt,) vec6i
        """
        if self.Topo_np is None:
            raise RuntimeError("Call load_gmsh() before build_edges().")

        topo = self.Topo_np
        nt = topo.shape[0]

        local_edges = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 3],
            [2, 3],
        ], dtype=np.int32)

        all_edges = []
        for e0, e1 in local_edges:
            edges = np.sort(topo[:, [e0, e1]], axis=1)
            all_edges.append(edges)

        E_all = np.vstack(all_edges)                    # (6*nt, 2)
        E = np.unique(E_all, axis=0).astype(np.int32)
        E = np.ascontiguousarray(E)

        self.ne = E.shape[0]

        edge_id = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}

        Et = np.zeros((nt, 6), dtype=np.int32)

        for t in range(nt):
            tet = topo[t]
            for loc, (i, j) in enumerate(local_edges):
                a = int(tet[i])
                b = int(tet[j])
                key = tuple(sorted((a, b)))
                idx = edge_id[key]

                # signed edge orientation relative to local ordering
                sigma = 1 if a < b else -1
                Et[t, loc] = sigma * idx

        self.E_np = E
        self.Et_np = np.ascontiguousarray(Et)
        self._upload_edge_arrays()

        return self.E, self.Et

    # =========================================================
    # Build unique faces
    # =========================================================
    def build_faces(self):
        """
        Each tetra has 4 triangular faces.

        Local faces:
            face 0 opposite vertex 0: (1,2,3)
            face 1 opposite vertex 1: (0,3,2)
            face 2 opposite vertex 2: (0,1,3)
            face 3 opposite vertex 3: (0,2,1)

        For uniqueness, store sorted vertex ids in self.F.
        self.Ft stores face ids for each tetra.
        """
        if self.Topo_np is None:
            raise RuntimeError("Call load_gmsh() before build_faces().")

        topo = self.Topo_np
        nt = topo.shape[0]

        local_faces = [
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (0, 2, 1),
        ]

        all_faces = []
        for f in local_faces:
            face = np.sort(topo[:, list(f)], axis=1)
            all_faces.append(face)

        F_all = np.vstack(all_faces)                    # (4*nt, 3)
        F = np.unique(F_all, axis=0).astype(np.int32)
        F = np.ascontiguousarray(F)

        self.nf = F.shape[0]

        face_id = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(F)}

        Ft = np.zeros((nt, 4), dtype=np.int32)

        for t in range(nt):
            tet = topo[t]
            for loc, f in enumerate(local_faces):
                a, b, c = map(int, tet[list(f)])
                key = tuple(sorted((a, b, c)))
                Ft[t, loc] = face_id[key]

        self.F_np = F
        self.Ft_np = np.ascontiguousarray(Ft)
        self._upload_face_arrays()

        return self.F, self.Ft

    # =========================================================
    # Build shared / boundary faces
    # =========================================================
    def build_shared_faces(self):
        """
        In tetrahedral meshes, face adjacency is more important than edge adjacency.

        Output:
            self.shared_face_idx      : faces shared by exactly 2 tetrahedra
            self.face_to_tets         : the two incident tetra ids for each shared face
            self.boundary_face_idx    : faces belonging to exactly 1 tetrahedron
            self.boundary_face_to_tet : incident tetra for each boundary face
            self.opposite_vertices    : for a shared face, the opposite vertex
                                        in each of the 2 tetrahedra
        """
        if self.F_np is None or self.Ft_np is None or self.Topo_np is None:
            raise RuntimeError("Call build_faces() before build_shared_faces().")

        F = self.F_np
        Ft = self.Ft_np
        Topo = self.Topo_np

        incident = [[] for _ in range(self.nf)]
        for t in range(self.nt):
            for loc in range(4):
                f = int(Ft[t, loc])
                incident[f].append(t)

        counts = np.array([len(v) for v in incident], dtype=np.int32)

        shared_face_idx = np.where(counts == 2)[0].astype(np.int32)
        boundary_face_idx = np.where(counts == 1)[0].astype(np.int32)

        if shared_face_idx.shape[0] > 0:
            face_to_tets = np.array(
                [incident[f] for f in shared_face_idx], dtype=np.int32
            ).reshape(-1, 2)
        else:
            face_to_tets = np.zeros((0, 2), dtype=np.int32)

        if boundary_face_idx.shape[0] > 0:
            boundary_face_to_tet = np.array(
                [incident[f][0] for f in boundary_face_idx], dtype=np.int32
            )
        else:
            boundary_face_to_tet = np.zeros((0,), dtype=np.int32)

        opposite_vertices = np.empty((shared_face_idx.shape[0], 2), dtype=np.int32)
        for r, f in enumerate(shared_face_idx):
            t1, t2 = face_to_tets[r]
            face_set = set(map(int, F[f]))

            opp1 = list(set(map(int, Topo[t1])) - face_set)
            opp2 = list(set(map(int, Topo[t2])) - face_set)

            if len(opp1) != 1 or len(opp2) != 1:
                raise ValueError(
                    f"Failed to find unique opposite vertices for face {f}, "
                    f"shared by tetrahedra {t1} and {t2}."
                )

            opposite_vertices[r, 0] = opp1[0]
            opposite_vertices[r, 1] = opp2[0]

        self.shared_face_idx_np = np.ascontiguousarray(shared_face_idx)
        self.face_to_tets_np = np.ascontiguousarray(face_to_tets)
        self.boundary_face_idx_np = np.ascontiguousarray(boundary_face_idx)
        self.boundary_face_to_tet_np = np.ascontiguousarray(boundary_face_to_tet)
        self.opposite_vertices_np = np.ascontiguousarray(opposite_vertices)

        self._upload_shared_face_arrays()

        return (
            self.shared_face_idx,
            self.face_to_tets,
            self.boundary_face_idx,
            self.boundary_face_to_tet,
            self.opposite_vertices,
        )

    ## Calculate the boundary vertices
    def get_surface_vertices_numpy(self):

        if self.boundary_face_idx_np is None or self.F_np is None:

            raise RuntimeError("Call build_shared_faces() first.")
        
        boundary_faces = self.F_np[self.boundary_face_idx_np]   # (nbf, 3)

        self.surface_vertices = np.unique(boundary_faces.reshape(-1))

        return

    def get_oriented_boundary_faces_numpy(self):
        if self.Topo_np is None:
            raise RuntimeError("Call load_gmsh() first.")

        topo = self.Topo_np

        # Local face definitions for a tetrahedron, with orientation:
        local_faces = [
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (0, 2, 1),
        ]

        face_map = {}

        for t in range(self.nt):
            tet = topo[t]
            for loc, f in enumerate(local_faces):
                oriented_face = tuple(int(tet[i]) for i in f)
                key = tuple(sorted(oriented_face))

                if key in face_map:
                    # if the face is already in the map, it means we've seen it before with the opposite orientation
                    del face_map[key]
                else:
                    # First time seeing this face, add it to the map
                    face_map[key] = oriented_face

        boundary_faces = np.array(list(face_map.values()), dtype=np.int32)
        return np.ascontiguousarray(boundary_faces)

    def summary(self):
        print("========== Mesh Summary ==========")
        print(f"#vertices      : {self.nv}")
        print(f"#tetrahedra    : {self.nt}")
        print(f"#unique edges  : {self.ne}")
        print(f"#unique faces  : {self.nf}")

        if self.shared_face_idx_np is not None:
            print(f"#shared faces  : {self.shared_face_idx_np.shape[0]}")
        if self.boundary_face_idx_np is not None:
            print(f"#boundary faces: {self.boundary_face_idx_np.shape[0]}")
        print(f"device         : {self.device}")
        print("==================================")