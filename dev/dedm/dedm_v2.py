import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from typing import Dict

# TODO retrain

# --- Core E3NN Graph Layer (No change) ---
class E3GNNConv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_edge_attr, num_neighbors):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors
        self.tp = FullyConnectedTensorProduct(irreps_in, irreps_edge_attr, self.irreps_out, shared_weights=True, internal_weights=True)
        self.fc = FullyConnectedNet([10, 64, 1], torch.nn.functional.silu)
        
    def forward(self, node_in, edge_src, edge_dst, edge_attr, edge_len_emb):
        messages_angular = self.tp(node_in[edge_src], edge_attr)
        messages_radial = self.fc(edge_len_emb)
        messages = messages_angular * messages_radial
        agg_messages = torch_scatter_sum(messages, edge_dst, dim=0, dim_size=node_in.shape[0])
        return agg_messages / (self.num_neighbors**0.5)

# --- Main Simplified Network (Fix Applied Here) ---
class SimpleEquivariantNetwork(torch.nn.Module):
    def __init__(self, irreps_node_in, lmax, **kwargs):
        super().__init__()
        self.max_radius = kwargs.get("max_radius", 5.0)
        self.lmax = lmax
        self.num_neighbors = kwargs.get("num_neighbors_avg", 15.0)
        hidden_mul = kwargs.get("hidden_mul", 64)
        num_layers = kwargs.get("num_layers", 3)

        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        irreps_node_hidden = o3.Irreps([(hidden_mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]).simplify()
        
        # --- FIX: Add an initial embedding layer ---
        # This layer projects the input features (dim 5) into the hidden dimension (dim 1152)
        self.embedding_layer = o3.Linear(irreps_node_in, irreps_node_hidden)
        
        self.encoder = torch.nn.ModuleList()
        # The encoder loop now consistently uses the hidden representation
        for _ in range(num_layers):
            conv = E3GNNConv(irreps_node_hidden, irreps_node_hidden, self.irreps_edge_attr, self.num_neighbors)
            self.encoder.append(conv)
            
        self.irreps_final_graph = irreps_node_hidden
        self.irreps_s2_out = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
        self.decoder = o3.Linear(self.irreps_final_graph, self.irreps_s2_out)

    def _orientation_to_cartesian(self, o): 
        theta, phi = o[..., 0], o[..., 1]
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)
    

    def forward(self, data):
        pos, node_features, batch = data["pos"], data["x"], data.get("batch", torch.zeros_like(data["pos"][:, 0], dtype=torch.long))
        
        edge_index = torch.cdist(pos, pos).le(self.max_radius).nonzero().T
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_len = edge_vec.norm(dim=1)
        mask = edge_len > 0
        edge_src, edge_dst, edge_vec, edge_len = edge_src[mask], edge_dst[mask], edge_vec[mask], edge_len[mask]
        
        edge_attr = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, 'component')
        edge_len_emb = soft_one_hot_linspace(edge_len, 0.0, self.max_radius, 10, "smooth_finite", True) * (10**0.5)
        
        # --- FIX: Apply the embedding layer before the loop ---
        h = self.embedding_layer(node_features)
        
        # This residual connection now works because the layer's input and output shapes match
        for layer in self.encoder: 
            h = h + layer(h, edge_src, edge_dst, edge_attr, edge_len_emb)
        
        graph_features = torch_scatter_mean(h, batch, dim=0)
        sh_coeffs = self.decoder(graph_features)
        query_vector = self._orientation_to_cartesian(data["orientation"])
        sh_query = o3.spherical_harmonics(self.irreps_s2_out, query_vector, False, 'component')
        return (sh_coeffs * sh_query).sum(dim=-1)


# --- Helper Functions (No change) ---
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

# --- Example Usage (No changes, should now run without error) ---
if __name__ == '__main__':
    # Setup for Scalar Output Network
    print("--- Testing Scalar Output Network (Corrected) ---")
    LMAX, num_nodes, batch_size = 2, 20, 2
    # data = {
    #     "pos": torch.randn(batch_size * num_nodes, 3),
    #     "x": torch.randn(batch_size * num_nodes, 1),
    #     "orientation": torch.randn(batch_size, 3),
    #     "batch": torch.arange(batch_size).repeat_interleave(num_nodes)
    # }
    from dataset.datasets import DragMeshDataset
    from torch_geometric.data import DataLoader

    ds = DragMeshDataset("/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/data/cube50k.dat", 
                     "/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/STLs/Cube_38_1m.stl", 
                     return_features_separately=False)
    dl = iter(DataLoader(ds, batch_size=batch_size, shuffle=False))
    data, y = next(dl)
    print(data)
    
    model = SimpleEquivariantNetwork(o3.Irreps("5x0e"), lmax=LMAX)
    output = model(data)
    print(f"Scalar Output shape: {output.shape}")
    assert output.shape == (batch_size,)

    # Setup for Vector Output Network
    # print("\n--- Testing Vector Output Network (Corrected) ---")
    # data.pop("orientation")
    # vec_model = SimpleEquivariantNetwork(o3.Irreps("1x0e"), lmax=LMAX)
    # vec_output = vec_model(data)
    # print(f"Simple Output shape: {vec_output.shape}")
    # assert vec_output.shape == (batch_size, 3)