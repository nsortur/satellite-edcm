import unittest
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data
from dedm_v2 import SimpleEquivariantNetwork, o3
from dedm import SimpleNetwork


# --- Test Suite ---
def get_cube_symmetry_rotations() -> list[torch.Tensor]:
    rotations = []
    rotations.append(R.from_matrix(np.eye(3)))
    for axis in np.eye(3):
        for angle in [90, 180, 270]:
            rotations.append(R.from_rotvec(axis * np.deg2rad(angle)))
    for axis in [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]:
        for angle in [120, 240]:
            rotations.append(R.from_rotvec(np.array(axis) / np.sqrt(3) * np.deg2rad(angle)))
    for axis in [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]:
        rotations.append(R.from_rotvec(np.array(axis) / np.sqrt(2) * np.deg2rad(180)))
    
    unique_matrices = {tuple(np.round(r.as_matrix(), 5).flatten()) for r in rotations}
    return [torch.tensor(np.array(m).reshape(3, 3), dtype=torch.float32) for m in unique_matrices]

def cartesian_to_spherical(xyz: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(xyz, p=2, dim=-1, keepdim=True)
    xyz_normalized = xyz / norm.clamp(min=1e-6)
    x, y, z = xyz_normalized.unbind(-1)
    theta = torch.acos(z.clamp(min=-1.0, max=1.0))
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=-1)


class TestModelProperties(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the model and data once for all tests."""
        cls.LMAX = 2
        
        # AssertionError: True is not false : Model output was unexpectedly invariant when only orientation changed.
        # cls.model = SimpleNetwork("5x0e", lmax=cls.LMAX, max_radius=1.7, num_neighbors=3.0, num_nodes=5.0,)
        
        # new model passes, fixed query vector rotation equivariance
        cls.model = SimpleEquivariantNetwork(o3.Irreps("5x0e"), lmax=cls.LMAX)
        cls.model.eval()

        pos = torch.tensor([
            [-1., -1., -1.], [-1., -1., 1.], [-1., 1., -1.], [-1., 1., 1.],
            [1., -1., -1.], [1., -1., 1.], [1., 1., -1.], [1., 1., 1.]
        ], dtype=torch.float32)
        
        x = torch.randn(pos.shape[0], 5) 
        orientation = torch.rand(1, 2) * torch.tensor([torch.pi, 2 * torch.pi])

        cls.data = Data(pos=pos, x=x, orientation=orientation, batch=torch.zeros(pos.shape[0], dtype=torch.long))
        cls.cube_symmetries = get_cube_symmetry_rotations()
        print(len(cls.cube_symmetries))
        
    def test_invariance_to_object_symmetry(self):
        """
        Tests H-Invariance.
        The output should be the same if we rotate the system by a symmetry
        operation `h` of the object. `f(h*pos, h*orient) == f(pos, orient)`.
        This verifies the proof from the original question.
        """
        with torch.no_grad():
            original_output = self.model(self.data)

        for i, R_h in enumerate(self.cube_symmetries):
            with self.subTest(f"Symmetry rotation {i+1}/{len(self.cube_symmetries)}"):
                rotated_data = self.data.clone()
                rotated_data.pos = self.data.pos @ R_h.T
                cart_orientation = self.model._orientation_to_cartesian(self.data.orientation)
                rotated_cart_orientation = cart_orientation @ R_h.T
                rotated_data.orientation = cartesian_to_spherical(rotated_cart_orientation)

                rotated_output = self.model(rotated_data)

                self.assertTrue(
                    torch.allclose(original_output, rotated_output, atol=1e-5),
                    f"Model failed H-invariance test for symmetry rotation {i+1}."
                )

    def test_invariance_to_global_rotation(self):
        """
        Tests SO(3)-Equivariance.
        The output must be the same if we rotate the ENTIRE system (mesh and
        orientation) by ANY random rotation `R`. This is because the relative
        configuration is unchanged.
        """
        with torch.no_grad():
            original_output = self.model(self.data)

        random_rotation = torch.tensor(R.random().as_matrix(), dtype=torch.float32)

        rotated_data = self.data.clone()
        rotated_data.pos = self.data.pos @ random_rotation.T
        cart_orientation = self.model._orientation_to_cartesian(self.data.orientation)
        rotated_cart_orientation = cart_orientation @ random_rotation.T
        rotated_data.orientation = cartesian_to_spherical(rotated_cart_orientation)

        rotated_output = self.model(rotated_data)

        self.assertTrue(
            torch.allclose(original_output, rotated_output, atol=1e-5),
            "Model failed global rotation invariance test. This would indicate a major model bug."
        )

    def test_dependence_on_orientation(self):
        """
        Tests that the output is NOT constant.
        The output MUST change if we change the relative orientation between the
        mesh and the query vector. This confirms the learned function is not trivial.
        """
        with torch.no_grad():
            original_output = self.model(self.data)

        random_rotation = torch.tensor(R.random().as_matrix(), dtype=torch.float32)

        # Create new data where ONLY the orientation is rotated.
        new_orientation_data = self.data.clone()
        cart_orientation = self.model._orientation_to_cartesian(self.data.orientation)
        rotated_cart_orientation = cart_orientation @ random_rotation.T
        new_orientation_data.orientation = cartesian_to_spherical(rotated_cart_orientation)
        
        with torch.no_grad():
            new_output = self.model(new_orientation_data)

        self.assertFalse(
            torch.allclose(original_output, new_output, atol=1e-4),
            "Model output was unexpectedly invariant when only orientation changed."
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
