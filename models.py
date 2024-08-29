import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn.nn as esnn
import escnn
from escnn import gspaces
from groups import GSpaceInfo
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_cluster import radius_graph
from torch_scatter import scatter
import e3nn.nn as enn

class DragMeshNetwork(nn.Module):
    def __init__(self, irreps_node_output, max_radius=3.2, 
                 lmax=3, num_basis_radial=10):
        super().__init__()
        self.num_basis_radial = num_basis_radial
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        # self.layers = nn.ModuleList()

        # both scalars
        irreps_scalars = "16x0e + 16x0o"
        irreps_gates = "8x0e + 8x0o + 8x0e + 8x0o"
        # gated tensors
        irreps_gated = "16x1o + 16x1e"

        gate = enn.Gate(irreps_scalars, [torch.relu, torch.tanh], 
                             irreps_gates, [torch.relu, torch.tanh, torch.relu, torch.tanh], 
                             irreps_gated)
        self.conv1 = GraphConv(self.irreps_sh, self.irreps_sh, gate.irreps_in, num_basis_radial=num_basis_radial)
        self.gate = gate
        self.conv2 = GraphConv(gate.irreps_out, self.irreps_sh, irreps_node_output, num_basis_radial=num_basis_radial)
        
        self.max_radius = max_radius
    
    def forward(self, data):
        num_nodes = data.x.shape[0]
        edge_src, edge_dst = radius_graph(data.pos.cpu(), self.max_radius, batch=data.batch)
        edge_src, edge_dst = edge_src.to(data.pos.device), edge_dst.to(data.pos.device)

        num_neighbors = len(edge_src) / num_nodes
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        edge_attr = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        edge_length_embedded = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.max_radius, 
                                    self.num_basis_radial, basis='smooth_finite', 
                                    cutoff=True).mul(self.num_basis_radial**0.5)
        
        # self.tp(f_in[edge_src], sh, self.fc(emb)) is initialized to edge_attr (spherical harmonics)
        x = scatter(edge_attr, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)
        x = self.conv1(x, edge_src, edge_dst, edge_attr, edge_length_embedded, num_neighbors)
        x = self.gate(x)
        x = self.conv2(x, edge_src, edge_dst, edge_attr, edge_length_embedded, num_neighbors)
        
        return scatter(x, data.batch, dim=0).div(num_nodes**0.5)

class GraphConv(nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_basis_radial) -> None:
        super().__init__()


        tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = enn.FullyConnectedNet([num_basis_radial, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars, num_neighbors) -> torch.Tensor:
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)
        return node_features

# class GraphConv(nn.Module):
#     def __init__(self, irreps_input, irreps_output, num_basis_radial, max_radius=1.8, sh_lmax=2):
#         super().__init__()
#         self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
#         self.irreps_input = irreps_input
#         self.irreps_output = irreps_output
#         self.max_radius = max_radius
#         self.num_basis_radial = num_basis_radial

#         self.tp = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_output, shared_weights=False)
#         self.fc = enn.FullyConnectedNet([num_basis_radial, 16, self.tp.weight_numel], torch.relu)
    
#     def forward(self, f_in, pos):
#         num_nodes = f_in.size(0)
#         edge_src, edge_dst = radius_graph(pos, self.max_radius, max_num_neighbors=len(pos) - 1)
#         num_neighbors = len(edge_src) / num_nodes
#         edge_vec = pos[edge_dst] - pos[edge_src]
#         sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
#         emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.max_radius, 
#                                     self.num_basis_radial, basis='smooth_finite', cutoff=True).mul(self.num_basis_radial**0.5)
#         return scatter(self.tp(f_in[edge_src], sh, self.fc(emb)), edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)


class InvariantDragMLP(nn.Module):
    def __init__(self, gspaceinfo: GSpaceInfo, hidden_dim=128, norm_features=False, norm_min=None, norm_max=None):
        act = gspaceinfo.group
        input_reps = gspaceinfo.input_reps

        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_features = norm_features

        self.norm_min = norm_min
        self.norm_max = norm_max

        self.network = esnn.SequentialModule(
             
            esnn.Linear(esnn.FieldType(act, 1 * input_reps),
                       esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.ReLU(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.Linear(esnn.FieldType(act, hidden_dim * [act.regular_repr]),
                       esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.ReLU(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.GroupPooling(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.Linear(esnn.FieldType(act, hidden_dim * [act.trivial_repr]),
                       esnn.FieldType(act, hidden_dim * [act.trivial_repr])),
        )
        
        self.network2 = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x): 
        dir_vec = x[:, 0:3].reshape((x.shape[0], 3))
        in_vars = x[:, 3:]
        if self.norm_features:
            in_vars = (in_vars-self.norm_min) / (self.norm_max - self.norm_min)
        
        geo_x = self.network.in_type(dir_vec)
        h = self.network(geo_x).tensor.reshape(x.shape[0], self.hidden_dim)
        cat = torch.cat((h, in_vars), axis=1)
        h2 = self.network2(cat)
        
        return h2
    

if __name__ == "__main__":
    from datasets import DragMeshDataset
    from torch_geometric.data import DataLoader

    ds = DragMeshDataset("data/cube50k.dat", "STLs/Cube_38_1m.stl")
    dl = iter(DataLoader(ds, batch_size=1, shuffle=False))
    network = DragMeshNetwork("1x0e")
    data = next(dl)
    out = network(data)
    print(out)
    
    rot_orientation = torch.tensor([0.1, 0.1])
    data_rot = ds._rotate(data, rot_orientation)
    print(network(data_rot))
    # model = TestVarToSphere()
    # test_inp = torch.rand((1, 2))
    # print(model(test_inp))
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D


    # create a 3d scatterplot
    
    # points = []
    # start_sample = [1, 2, 3]
    # points.append(start_sample)
    # act = gspaces.GSpace3D((False, "so3",))
    # for elem in act.fibergroup.testing_elements():
    #     transformed = act.fibergroup.standard_representation()(elem)@start_sample
    #     points.append(transformed)

    # x, y, z = zip(*points)
    # print(len(points))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x, y, z)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # plt.show()
