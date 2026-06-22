import numpy as np
from utils.MeshLoader import MeshLoader3D_Warp

class DisForceCalculator():
    def __init__(self, mesh: MeshLoader3D_Warp,
                 applied_idx,):
        self.mesh = mesh
        self.X = self.mesh.X_np
        self.surface = self.mesh.F_np
        self.applied_idx = applied_idx
        self.contact_faces = self.find_contact_faces_from_vertices()
        self.areas = np.array([
            self.triangle_area(self.X[f[0]], 
                               self.X[f[1]], 
                               self.X[f[2]])
            for f in self.contact_faces
        ], dtype=np.float32)
        self.total_area = np.sum(self.areas)

    def find_contact_faces_from_vertices(self):
        """
        surface_faces: (nf, 3) int, surface triangle vertex indices
        applied_idx: (n_selected,) int, selected surface vertices
        """
        applied_set = set(self.applied_idx.tolist())

        contact_faces = []
        for f in self.surface:
            if int(f[0]) in applied_set and int(f[1]) in applied_set and int(f[2]) in applied_set:
                contact_faces.append(f)

        return np.asarray(contact_faces, dtype=np.int32)


    def triangle_area(self, x0, x1, x2):
        return 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0))


    def distribute_force_to_contact_faces(self, total_force):
        """
        X_np: (nv, 3) vertex positions
        surface_faces: (nf, 3) surface triangle indices
        applied_idx: selected surface vertex indices
        total_force: (3,) total applied force
        """
        vertex_force = np.zeros_like(self.X, dtype=np.float32)

        if len(self.contact_faces) == 0:
            print("Warning: no contact faces found. Try larger AABB or allow partial faces.")
            return vertex_force, self.contact_faces

        if self.total_area < 1e-12:
            print("Warning: contact area is nearly zero.")
            return vertex_force, self.contact_faces

        total_force = np.asarray(total_force, dtype=np.float32)

        for f, area in zip(self.contact_faces, self.areas):
            face_force = total_force * (area / self.total_area)

            # linear triangle: equally distribute face force to 3 nodes
            vertex_force[f[0]] += face_force / 3.0
            vertex_force[f[1]] += face_force / 3.0
            vertex_force[f[2]] += face_force / 3.0

        segmented_vertex_force = vertex_force[self.applied_idx]

        return segmented_vertex_force