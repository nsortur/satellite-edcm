from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
import torch


class ExampleWrapper(torch.nn.Module):
    """Example wrapper around a GATr model.
    
    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.
    
    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(self, blocks=20, hidden_mv_channels=16, hidden_s_channels=32):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        
    def forward(self, inputs):
        """Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data
        
        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        
        # Embed point cloud in PGA
        embedded_inputs = embed_point(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)
        
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=None)  # (..., num_points, 1, 16)
        
        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        outputs = torch.mean(nodewise_outputs, dim=(-3, -2))  # (..., 1)
        
        return outputs
    
    
if __name__ == "__main__":
    # Since the model outputs a single scalar, we test for rotation invariance.
    # A scalar output should not change when the input point cloud is rotated.

    # For reproducibility
    torch.manual_seed(42)

    # Instantiate a smaller model for a quick test
    model = ExampleWrapper(blocks=2, hidden_mv_channels=4, hidden_s_channels=8)

    # Create a random point cloud
    # (batch_size, num_points, 3)
    points = torch.randn(1, 10, 3)

    # Create a random rotation matrix (e.g., 45 degrees around z-axis)
    angle = torch.tensor(torch.pi / 4.0)
    c, s = torch.cos(angle), torch.sin(angle)
    rotation_matrix = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=points.dtype)

    # Apply the rotation to the point cloud
    # (1, 10, 3) @ (3, 3) -> (1, 10, 3)
    rotated_points = points @ rotation_matrix.T

    # Pass both original and rotated point clouds through the model
    output_original = model(points)
    output_rotated = model(rotated_points)

    print(f"Output (original): {output_original.item():.6f}")
    print(f"Output (rotated):  {output_rotated.item():.6f}")

    # Check if the outputs are nearly identical
    is_invariant = torch.allclose(output_original, output_rotated, atol=1e-6)
    print(f"\nModel is rotationally invariant: {is_invariant}")

    assert is_invariant, "Invariance test failed: outputs are not the same after rotation."
    print("Invariance test passed successfully.")
    