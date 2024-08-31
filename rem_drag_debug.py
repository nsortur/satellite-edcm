from typing import Dict, Union

import torch
import torch as tr
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate

import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
import math


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class MessagePassing(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features

    irreps_node_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_node_output : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    layers : int
        number of gates (non linearities)

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_hidden,
        irreps_node_output,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        irreps_node = self.irreps_node_input

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, fc_neurons, num_neighbors
            )
            irreps_node = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, self.irreps_node_output, fc_neurons, num_neighbors
            )
        )

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features


def test() -> None:
    from torch_cluster import radius_graph
    from e3nn.util.test import assert_equivariant, assert_auto_jitable

    mp = MessagePassing(
        irreps_node_input="0e",
        irreps_node_hidden="0e + 1e",
        irreps_node_output="1e",
        irreps_node_attr="0e + 1e",
        irreps_edge_attr="1e",
        layers=3,
        fc_neurons=[2, 100],
        num_neighbors=3.0,
    )

    num_nodes = 4
    node_pos = torch.randn(num_nodes, 3)
    edge_index = radius_graph(node_pos, 3.0)
    edge_src, edge_dst = edge_index
    num_edges = edge_index.shape[1]
    edge_attr = node_pos[edge_index[0]] - node_pos[edge_index[1]]

    node_features = torch.randn(num_nodes, 1)
    node_attr = torch.randn(num_nodes, 4)
    edge_scalars = torch.randn(num_edges, 2)

    assert mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars).shape == (num_nodes, 3)

    assert_equivariant(
        mp,
        irreps_in=[mp.irreps_node_input, mp.irreps_node_attr, None, None, mp.irreps_edge_attr, None],
        args_in=[node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars],
        irreps_out=[mp.irreps_node_output],
    )

    assert_auto_jitable(mp.layers[0].first)


def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

# L^2 * SO(3) -> natural decomposition for a signal over SO(3)
def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


class SO3ToS2Convolution(torch.nn.Module):
  def __init__(self, f_in, f_out, lmax_in, lmax_out, kernel_grid):
    super().__init__()
    self.lin = o3.Linear(so3_irreps(lmax_in), s2_irreps(lmax_out), f_in=f_in, f_out=f_out)

  def forward(self, x):
    return self.lin(x)
  

def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3):
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))
  

class SimpleNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        max_radius,
        num_neighbors: int,
        num_nodes: int,
        mul: int = 50,
        layers: int = 3,
        lmax: int = 2,
        pool_nodes: bool = True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.mp_irreps_in = self.mp.irreps_node_input
        # self.irreps_out = self.mp.irreps_node_output
        grid_s2 = s2_near_identity_grid()
        self.lin = o3.Linear(self.mp.irreps_node_output, so3_irreps(lmax), f_in=128, f_out=128)
        self.so3tos2 = SO3ToS2Convolution(
            128, 56, lmax_in=lmax, lmax_out=lmax, kernel_grid=grid_s2
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data["pos"], self.max_radius, batch, max_num_neighbors=len(data["pos"]) - 1)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return batch, data["x"], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        # del data

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            pooled = scatter(node_outputs, batch, dim=0, reduce="mean")
        else:
            pooled = node_outputs

        batch_size = data.batch.max() + 1
        node_outputs = self.lin(pooled.view(batch_size, 1, -1))
        node_outputs = self.so3tos2(node_outputs)
        return node_outputs


def ar2los(x_ar):
    """Convert a unit spherical coordinate to cartesian.
    Parameters
    ----------
    x_ar: Tensor, shape-(N, ..., [2, 4, 6])
        Aspect/Roll coordinates
    Returns
    -------
    x_los: Tensor, shape-(N, ..., [3, 6, 9])
        Cartesian coordinates
    """
    assert x_ar.shape[-1] % 2 == 0
    assert x_ar.shape[-1] <= 6

    # Line-of-sight in XYZ
    a = x_ar[..., 0]
    r = x_ar[..., 1]

    x = -tr.sin(a) * tr.cos(r)
    y = -tr.sin(a) * tr.sin(r)
    z = -tr.cos(a)

    if x_ar.shape[-1] == 2:
        return tr.stack([x, y, z], dim=-1)

    # First time derivative
    da_dt = x_ar[..., 2]
    dr_dt = x_ar[..., 3]

    # Non-zero partial derivatives
    dxlos_da = -tr.cos(a) * tr.cos(r)
    dxlos_dr = tr.sin(a) * tr.sin(r)
    dylos_da = -tr.cos(a) * tr.sin(r)
    dylos_dr = -tr.sin(a) * tr.cos(r)
    dzlos_da = tr.sin(a)

    # Time derivative of line-of-sight
    xd = dxlos_da * da_dt + dxlos_dr * dr_dt
    yd = dylos_da * da_dt + dylos_dr * dr_dt
    zd = dzlos_da * da_dt

    if x_ar.shape[-1] == 4:
        return tr.stack([x, y, z, xd, yd, zd], dim=-1)

    da_dtdt = x_ar[..., 4]
    dr_dtdt = x_ar[..., 5]

    # Second partial derivatives
    dxlos_dada = tr.sin(a) * tr.cos(r)
    dxlos_dadr = tr.cos(a) * tr.sin(r)
    dxlos_drda = tr.cos(a) * tr.sin(r)
    dxlos_drdr = tr.sin(a) * tr.cos(r)
    dylos_dada = tr.sin(a) * tr.sin(r)
    dylos_dadr = -tr.cos(a) * tr.cos(r)
    dylos_drda = -tr.cos(a) * tr.cos(r)
    dylos_drdr = tr.sin(a) * tr.sin(r)
    dzlos_dada = tr.cos(a)

    # Second time derivative of line-of-sight
    xdd = (
        (dxlos_dada * da_dt + dxlos_dadr * dr_dt) * da_dt
        + dxlos_da * da_dtdt
        + (dxlos_drda * da_dt + dxlos_drdr * dr_dt) * dr_dt
        + dxlos_dr * dr_dtdt
    )
    ydd = (
        (dylos_dada * da_dt + dylos_dadr * dr_dt) * da_dt
        + dylos_da * da_dtdt
        + (dylos_drda * da_dt + dylos_drdr * dr_dt) * dr_dt
        + dylos_dr * dr_dtdt
    )
    zdd = (dzlos_dada * da_dt) * da_dt + dzlos_da * da_dtdt

    return tr.stack([x, y, z, xd, yd, zd, xdd, ydd, zdd], dim=-1)


def test_simple_network() -> None:
    from rem import REM
    from datasets import DragMeshDataset
    from torch_geometric.data import DataLoader
    from e3nn.io import SphericalTensor
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # net = REM(num_node_features=5, z_lmax=4, max_radius=1.8, out_dim=1)
    
    # # GRACE_A_tpmc
    # ds = DragMeshDataset("data/cube50k.dat", "STLs/GRACE_A_tpmc.stl")
    # dl = iter(DataLoader(ds, batch_size=1, shuffle=False))
    # out = net(next(dl)[0])
    # aspects = torch.linspace(0, torch.pi, 360)
    # rolls = torch.zeros_like(aspects)
    # ar = torch.stack([aspects, rolls], -1)
    # print("ar", ar.shape)
    # los = ar2los(ar)
    # print("los", los.shape)
    # print("out", out.shape)

    # attitude_query = los[100:101, :]
    # print(attitude_query.shape)
    # res = net.getResponse(out, attitude_query)
    # print("res", res.item(), attitude_query)

    # rot90_attitude_query = attitude_query @ o3.Irrep("1e").D_from_angles(torch.tensor(math.pi), tr.tensor(0), tr.tensor(0))
    # res = net.getResponse(out, rot90_attitude_query)
    # print("res rot90", res.item(), rot90_attitude_query)

    # random_attitude_query = attitude_query @ o3.Irrep("1e").D_from_matrix(o3.rand_matrix())
    # # print("Random attitude", random_attitude_query)
    # res = net.getResponse(out, random_attitude_query)
    # print("res random", res.item(), random_attitude_query)

    # # irreps of spherical harmonics
    # x = SphericalTensor(4, 1, -1)
    # traces = x.plotly_surface(out.detach().squeeze())
    # traces = [go.Surface(**d) for d in traces]
    # fig = go.Figure(data=traces)
    # fig.show()

    ds = DragMeshDataset("data/cube50k.dat", "STLs/Cube_38_1m.stl")
    dl = iter(DataLoader(ds, batch_size=1, shuffle=False))
    network = REM(5, 4, 1.8, 1)
    data = next(dl)
    out = network(data[0])
    print("Pred", out)
    print("Target", data[1])

if __name__ == "__main__":
    test_simple_network()
    