import warp as wp
import numpy as np
import json
from utils.MeshLoader import MeshLoader3D_Warp

class KeyPointsBarrycentric:
    def __init__(
        self,
        keypoints_path: str,
        mesh_loader: MeshLoader3D_Warp,
        wp_device: wp.Device = None,
    ):
        with open(keypoints_path, "r") as f:
            keypoints = json.load(f)

        self.keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
        self.kp_num = len(self.keypoints)
        self.mesh = mesh_loader
        self.wp_device = wp_device

        # Convert Warp arrays to NumPy if needed
        self.X = self.mesh.X_np
        self.Topo = self.mesh.Topo_np

        self.X = np.asarray(self.X, dtype=np.float32)
        self.Topo = np.asarray(self.Topo, dtype=np.int32)

        self.keypoints_tet_lis = np.full(self.kp_num, -1, dtype=np.int32)
        self.keypoints_barycentric_matrix = np.zeros((self.kp_num, 4), dtype=np.float32)

        self.compute_barycentric_coordinates()

        self.keypoints_tet_wp = wp.array(
            self.keypoints_tet_lis,
            dtype=wp.int32,
            device=wp_device,
        )

        self.keypoints_barycentric_wp = wp.array(
            self.keypoints_barycentric_matrix,
            dtype=wp.vec4,
            device=wp_device,
        )

    def compute_barycentric_coordinates(self):
        for k, kp in enumerate(self.keypoints):
            tet_id, weights = self.find_containing_or_closest_tet(kp)

            self.keypoints_tet_lis[k] = tet_id
            self.keypoints_barycentric_matrix[k] = weights

            print(
                f"Keypoint {k}: tet = {tet_id}, "
                f"weights = {weights}, sum = {np.sum(weights):.6f}"
            )

    def compute_barycentric_weights(self, p, tet_vertices):
        """
        tet_vertices order:
        [v0, v1, v2, v3]

        Return weights:
        [w0, w1, w2, w3]
        such that:
        p = w0*v0 + w1*v1 + w2*v2 + w3*v3
        """
        v0, v1, v2, v3 = tet_vertices

        D = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        rhs = p - v0

        try:
            w123 = np.linalg.solve(D, rhs)
        except np.linalg.LinAlgError:
            return None

        w1, w2, w3 = w123
        w0 = 1.0 - w1 - w2 - w3

        return np.array([w0, w1, w2, w3], dtype=np.float32)
    
    def is_inside_tet(self, weights, eps=1e-6):
        return np.all(weights >= -eps) and np.all(weights <= 1.0 + eps)

    def find_containing_or_closest_tet(self, kp):
        best_tet_id = -1
        best_weights = None
        best_score = np.inf

        for t in range(len(self.Topo)):
            tet_idx = self.Topo[t]
            tet_vertices = self.X[tet_idx]

            weights = self.compute_barycentric_weights(kp, tet_vertices)
            if weights is None:
                continue

            # Best case: keypoint is inside this tet
            if self.is_inside_tet(weights):
                return t, weights

            # Otherwise choose the tet whose barycentric weights are closest to valid range
            violation = np.sum(np.maximum(-weights, 0.0)) + np.sum(np.maximum(weights - 1.0, 0.0))

            if violation < best_score:
                best_score = violation
                best_tet_id = t
                best_weights = weights

        if best_tet_id == -1:
            raise RuntimeError("No valid tetrahedron found for keypoint.")

        return best_tet_id, best_weights