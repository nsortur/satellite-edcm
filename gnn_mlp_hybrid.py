from rem.rem import *

class GNN_MLP_Hybrid(tr.nn.Module):
    def __init__(self, num_node_features, num_inp_feats, mlp_hidden, z_lmax, max_radius, out_dim_gnn, out_dim_mlp):
        super(GNN_MLP_Hybrid, self).__init__()
        self.lmax = z_lmax
        f = out_dim_gnn
        self.num_inp_feats = num_inp_feats

        self.irreps_in = o3.Irreps(f"{num_node_features}x0e")
        # self.irreps_latent = e3nn_utils.so3_irreps(z_lmax)
        self.irreps_latent = o3.Irreps("1x0o")
        self.irreps_enc_out = o3.Irreps(
            [(f, (l, p)) for l in range((z_lmax) + 1) for p in [-1,1]]
        )

        self.encoder = GNN(
            irreps_node_input=self.irreps_in,
            irreps_node_output=self.irreps_enc_out,
            max_radius=max_radius,
            layers=2,
            mul=f,
            #lmax=[self.lmax // 2, self.lmax // 2, self.lmax // 2],
            lmax=[self.lmax, self.lmax, self.lmax],
        )
        self.lin = o3.Linear(self.irreps_enc_out, self.irreps_latent, f_in=1, f_out=f)
        self.mlp = nn.ModuleList([nn.Linear(num_inp_feats + f, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, out_dim_mlp)])
        
    def forward(self, inp_graph):
        B = inp_graph.batch.max() + 1
        feats = inp_graph.feats
        batch_size = inp_graph.batch.max() + 1

        pos = inp_graph.pos
        ip = o3.Irreps("1x1o")
        rot = ip.D_from_angles(inp_graph.orientation[0, 0], inp_graph.orientation[0, 1], tr.tensor(0))
        inp_graph.pos = pos @ rot.squeeze()

        gnn_out = self.encoder(inp_graph)
        z_gnn = self.lin(gnn_out.view(batch_size, 1, -1)) # B x 1 x f
        feats = feats.unsqueeze(0).reshape(B, self.num_inp_feats)
        cat = torch.cat((z_gnn.squeeze(2), feats), dim=1)
        for layer in self.mlp:
            cat = layer(cat)
        
        return cat

        return None
    
if __name__ == "__main__":
    from datasets import DragMeshDataset
    from torch_geometric.data import DataLoader

    # hyperparameter search, with and without cross attention
    net = GNN_MLP_Hybrid(num_node_features=5, num_inp_feats=5, mlp_hidden=128, z_lmax=4, max_radius=1.8, out_dim_gnn=32, out_dim_mlp=1)
    
    # GRACE_A_tpmc
    ds = DragMeshDataset("data/cube50k.dat", "STLs/Cube_38_1m.stl", return_features_separately=True)
    dl = iter(DataLoader(ds, batch_size=1, shuffle=False))
    inp_graph, y = next(dl)
    out = net(inp_graph)
    print(out)
