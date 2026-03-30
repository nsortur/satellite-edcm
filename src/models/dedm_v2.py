import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from torch_geometric.data import Data, Batch
from torch_cluster import radius_graph
from torch_geometric.utils import scatter
from torch.utils.checkpoint import checkpoint


# --- Custom Scatter Functions ---
def torch_scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    index, src_shape = index.unsqueeze(-1).expand_as(src), src.shape
    dim = dim if dim >= 0 else len(src_shape) + dim
    if out is None:
        size = list(src_shape); size[dim] = int(index.max()) + 1 if dim_size is None else dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

def torch_scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = torch_scatter_sum(src, index, dim, out, dim_size)
    count = torch_scatter_sum(torch.ones_like(src), index, dim, None, dim_size)
    return out / count.clamp(min=1)


# --- Core E3NN Graph Layer (Dev Version) ---
class E3GNNConv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_edge_attr, num_neighbors):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors
        
        # internal_weights=False so we can provide our own from the MLP
        self.tp = FullyConnectedTensorProduct(
            irreps_in, 
            irreps_edge_attr, 
            self.irreps_out, 
            shared_weights=False, 
            internal_weights=False
        )
        
        # The MLP output size matches the number of weights in the TP
        self.fc = FullyConnectedNet(
            [10, 64, self.tp.weight_numel], 
            torch.nn.functional.silu
        )
        
    def forward(self, node_in, edge_src, edge_dst, edge_attr, edge_len_emb):
        # 1. Generate the interaction weights from the distance embedding
        weight = self.fc(edge_len_emb)
        
        # 2. Perform the Tensor Product using those specific weights
        messages = self.tp(node_in[edge_src], edge_attr, weight)
        
        # 3. Aggregate
        agg_messages = scatter(messages, edge_dst, dim=0, dim_size=node_in.shape[0], reduce='sum')
        
        # 4. Normalize by the expected number of neighbors
        return agg_messages / (self.num_neighbors**0.5)


# --- Main Network (Replaces old DEDMV2) ---
class DEDMV2(torch.nn.Module):
    """
    Equivariant Drag Model V2 (Integrated from Dev SimpleEquivariantNetwork)
    """
    def __init__(self, irreps_node_in, lmax, **kwargs):
        super().__init__()
        self.max_radius = kwargs.get("max_radius", 2.0)
        self.lmax = lmax
        self.num_neighbors = kwargs.get("num_neighbors_avg", 15.0)
        hidden_mul = kwargs.get("hidden_mul", 64)
        num_layers = kwargs.get("num_layers", 3)

        # 1. Define Irreps for Gating
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        
        # irreps_gated = o3.Irreps([(hidden_mul, (l, p)) for l in range(1, lmax + 1) for p in [-1, 1]]).simplify()
        irreps_gated = o3.Irreps([(hidden_mul, (l, 1 if l % 2 == 0 else -1)) for l in range(1, lmax + 1)]).simplify()
        num_gated_channels = irreps_gated.num_irreps
        irreps_gates = o3.Irreps([(num_gated_channels, (0, 1))]) 

        # 3. Define the scalars that just get activated (L=0)
        irreps_scalars = o3.Irreps([(hidden_mul, (0, 1))])
        
        # This is what comes OUT of the gate
        self.irreps_node_hidden = (irreps_scalars + irreps_gated).simplify()
        
        # This is what the CONVOLUTION must produce to feed the gate
        irreps_conv_out = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        # 2. Modules
        self.embedding_layer = o3.Linear(irreps_node_in, self.irreps_node_hidden)
        
        self.encoder = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            # Convolution outputs the 'fat' irreps (including the gate scalars)
            conv = E3GNNConv(self.irreps_node_hidden, irreps_conv_out, self.irreps_edge_attr, self.num_neighbors)
            self.encoder.append(conv)
            
            # Gate processes the fat irreps back into the slim hidden irreps
            gate = Gate(
                irreps_scalars, [torch.nn.functional.silu],
                irreps_gates,   [torch.nn.functional.sigmoid],
                irreps_gated
            )
            self.gates.append(gate)
            
        self.irreps_final_graph = self.irreps_node_hidden
        self.irreps_s2_out = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
        self.decoder = o3.Linear(self.irreps_final_graph, self.irreps_s2_out)

    def _orientation_to_cartesian(self, o): 
        theta, phi = o[..., 0], o[..., 1]
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)

    def forward(self, data):
        # Extract properties dynamically from PyG Data/Batch object
        pos = data.pos
        node_features = data.x
        
        # Handle batch index (defaults to 0 if passing a single graph)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros_like(pos[:, 0], dtype=torch.long)
        
        # --- Graph Construction ---
        # Optimization: Use precomputed edge_index if it exists in the dataset, otherwise compute it
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
        else:
            edge_index = radius_graph(pos, r=self.max_radius, batch=batch, loop=False, max_num_neighbors=4)
            edge_src, edge_dst = edge_index[0], edge_index[1]
            
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_len = edge_vec.norm(dim=1)
        
        # Filter out self-loops or degenerate edges
        mask = edge_len > 1e-8 
        edge_src, edge_dst, edge_vec, edge_len = edge_src[mask], edge_dst[mask], edge_vec[mask], edge_len[mask]
        
        # --- Geometric Features ---
        edge_attr = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, 'component')
        edge_len_emb = soft_one_hot_linspace(edge_len, 0.0, self.max_radius, 10, "smooth_finite", True) * (10**0.5)
        envelope = 0.5 * (torch.cos(edge_len * 3.141592653589793 / self.max_radius) + 1.0)
        edge_len_emb = edge_len_emb * envelope.unsqueeze(-1)

        # --- Message Passing with Gating ---
        h = self.embedding_layer(node_features)
        for conv, gate in zip(self.encoder, self.gates):
            
            # Pass the conv module directly and use the reentrant=True engine
            res = checkpoint(
                conv, 
                h, edge_src, edge_dst, edge_attr, edge_len_emb, 
                use_reentrant=True
            )
            
            h = h + gate(res) 
            # res = conv(h, edge_src, edge_dst, edge_attr, edge_len_emb)
            # h = h + gate(res)
        
        # --- Pooling and Output ---
        graph_features = torch_scatter_mean(h, batch, dim=0)
        sh_coeffs = self.decoder(graph_features)
        
        query_vector = self._orientation_to_cartesian(data.orientation)
        sh_query = o3.spherical_harmonics(self.irreps_s2_out, query_vector, True, 'component')
        
        return (sh_coeffs * sh_query).sum(dim=-1)


# --- Example Usage & Validation ---
if __name__ == '__main__':
    # Try importing the new preprocessed datamodule
    try:
        from preprocessed_data_module import PreprocessedDragDataModule
        
        print("--- Testing DEDMV2 with Preprocessed Dataset ---")
        
        # Initialize DataModule
        dm = PreprocessedDragDataModule(
            data_dir="/projects/gllab/sortur.n/drag_dataset_84k.pt",
            batch_size=4,
            num_workers=0
        )
        dm.setup()
        
        # Get one real batch
        batch = next(iter(dm.train_dataloader()))
        
        # Note: irreps_node_in="8x0e" matches the 8 normalized physical features 
        # (Vx, Vy, Vz, Ox, Oy, Oz, Acc, Temp)
        model = DEDMV2(irreps_node_in="8x0e", lmax=2, max_radius=2.0, hidden_mul=32)
        model.eval()

        # Forward Pass
        output = model(batch)
        print(f"Batch targets (y) shape: {batch.y.shape}")
        print(f"Model output shape:      {output.shape}")
        print(f"Sample Outputs:          {output.detach().numpy()}")
        
        assert output.shape == batch.y.squeeze().shape or output.shape == batch.y.shape, "Output shape mismatch!"
        print("✅ Forward pass successful!")

    except ImportError:
        print("Could not import PreprocessedDragDataModule. Make sure it is in the same directory.")
    except FileNotFoundError:
        print("Could not find the drag_dataset_84k.pt file to test against.")
