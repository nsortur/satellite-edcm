from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, extract_oriented_plane
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
        
        nodewise_outputs = extract_point(embedded_outputs)
        # nodewise_outputs = extract_oriented_plane(embedded_outputs)
        return nodewise_outputs
    
    
if __name__ == "__main__":
    
    points = torch.randn(1, 5, 3)
    wrapper = ExampleWrapper(blocks=2, hidden_mv_channels=4, hidden_s_channels=8)
    
    angle = torch.tensor(torch.pi / 8.0)
    c, s = torch.cos(angle), torch.sin(angle)
    g = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=points.dtype)
    
    # g(f(x))
    nodewise_outputs = wrapper(points)
    g_f_x = nodewise_outputs.squeeze() @ g.T
    print(g_f_x)
    print(g_f_x.shape)
    
    # f(g(x))
    g_x = points @ g.T
    nosewise_outputs_gx = wrapper(g_x).squeeze()
    print(nosewise_outputs_gx)
    print(nosewise_outputs_gx.shape)
    
    print(torch.allclose(nosewise_outputs_gx, g_f_x))
   
    # in the self.gatr call, that's where self.scalar values will go (atmosphere/satellite temps, coefficients, etc.)
    # sec 3.3 of the paper
    
    # what is the action on the inputs and outputs of the drag problem? remember the C4 reacher rotation, how the action rotates
    
    