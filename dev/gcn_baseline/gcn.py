import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, use_gdc=False):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels_list[0], normalize=not use_gdc))
        for i in range(1, len(hidden_channels_list)):
            self.convs.append(GCNConv(hidden_channels_list[i-1], hidden_channels_list[i], normalize=not use_gdc))
        self.convs.append(GCNConv(hidden_channels_list[-1], out_channels, normalize=not use_gdc))

    def forward(self, data, edge_weight=None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)

        x = global_mean_pool(x, batch).squeeze(-1)
        return x

if __name__ == "__main__":
    from dataset.datasets import DragMeshDataset
    from torch_geometric.data import DataLoader

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    use_gdc = True
    if use_gdc:
        transform = "gdc_transform"
    else:
        transform = None
    
    ds = DragMeshDataset("/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/data/cube50k.dat", 
                     "/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/STLs/Cube_38_1m.stl", 
                     orientation_features=True, transform="gdc_transform")
    dl = iter(DataLoader(ds, batch_size=16, shuffle=False))
    
    model = GCN(
        in_channels=7,
        hidden_channels_list=[16, 16],
        out_channels=1,
        use_gdc=use_gdc,
    ).to(device)

    model = model.eval()
    data, y = next(dl)
    data = data.to(device)
    out = model(data)
    
    print(y, out)
    print(y.shape)
    print(out.shape)
