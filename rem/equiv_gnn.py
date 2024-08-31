from typing import Dict

import torch
import torch_geometric.utils
from e3nn import o3
from e3nn import nn as enn
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode

from . import e3nn_utils
from . import pooling


class GNN(torch.nn.Module):
  def __init__(
    self,
    irreps_node_input,
    irreps_node_output,
    max_radius,
    mul=50,
    layers=3,
    lmax=2,
    pool_nodes=True,
  ) -> None:
    super().__init__()

    self.lmax = lmax
    self.max_radius = max_radius
    self.number_of_basis = 10
    self.pool_nodes = pool_nodes

    irreps_node_hiddens = list()
    irreps_edge_hiddens = list()
    for layer_lmax in lmax[:-1]:
      irreps_node_hiddens.append(o3.Irreps(
        #[(mul * 2 * l + 1, (l, 1)) for l in range(layer_lmax + 1)]
        [(mul, (l, p)) for l in range(layer_lmax + 1) for p in [-1, 1]]
      ))
      irreps_edge_hiddens.append(o3.Irreps.spherical_harmonics(layer_lmax))


    irreps_node_seq = [irreps_node_input] + irreps_node_hiddens + [irreps_node_output]
    irreps_edge_seq = irreps_edge_hiddens + [o3.Irreps.spherical_harmonics(lmax[-1])]
    self.mp = MessagePassing(
      irreps_node_sequence=irreps_node_seq,
      irreps_edge_attrs=irreps_edge_seq,
      fc_neurons=[self.number_of_basis, 100],
      max_radius=max_radius,
      lmax=self.lmax
    )
    self.irreps_node_input = self.mp.irreps_node_input
    self.irreps_node_output = self.mp.irreps_node_output

  def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    batch, node_outputs = self.mp(data)

    if self.pool_nodes:
      return torch_geometric.utils.scatter(node_outputs, batch, dim=0, reduce='mean')
    else:
      return node_outputs

class MessagePassing(torch.nn.Module):
  def __init__(
    self,
    irreps_node_sequence,
    irreps_edge_attrs,
    fc_neurons,
    max_radius,
    lmax,
  ) -> None:
    super().__init__()
    self.lmax = lmax
    self.max_radius = max_radius
    self.number_of_basis = 10

    irreps_node_sequence = [o3.Irreps(irreps) for irreps in irreps_node_sequence]
    self.irreps_edge_attrs = [o3.Irreps(irreps) for irreps in irreps_edge_attrs]

    act = {
      1: torch.nn.functional.silu,
      -1: torch.tanh,
    }
    act_gates = {
      1: torch.sigmoid,
      -1: torch.tanh,
    }

    self.layers = torch.nn.ModuleList()

    self.irreps_node_sequence = [irreps_node_sequence[0]]
    irreps_node = irreps_node_sequence[0]

    for li, (irreps_node_hidden, irreps_edge_attr) in enumerate(zip(irreps_node_sequence[1:-1], self.irreps_edge_attrs[:-1])):
      irreps_scalars = o3.Irreps(
        [
          (mul, ir)
          for mul, ir in irreps_node_hidden
          if ir.l == 0
          and e3nn_utils.tp_path_exists(
            irreps_node, irreps_edge_attr, ir
          )
        ]
      ).simplify()
      irreps_gated = o3.Irreps(
        [
          (mul, ir)
          for mul, ir in irreps_node_hidden
          if ir.l > 0
          and e3nn_utils.tp_path_exists(
            irreps_node, irreps_edge_attr, ir
          )
        ]
      )
      if irreps_gated.dim > 0:
        if e3nn_utils.tp_path_exists(irreps_node, irreps_edge_attr, "0e"):
          ir = "0e"
        elif e3nn_utils.tp_path_exists(
          irreps_node, irreps_edge_attr, "0o"
        ):
          ir = "0o"
        else:
          raise ValueError(
            f"irreps_node={irreps_node} times irreps_edge_attr={self.irreps_edge_attr} is unable to produce gates "
            f"needed for irreps_gated={irreps_gated}"
          )
      else:
        ir = None
      irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

      gate = Gate(
        irreps_scalars,
        [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates,
        [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated,  # gated tensors
      )
      conv = GraphConvolution(
        irreps_node,
        irreps_edge_attr,
        gate.irreps_in,
        fc_neurons,
      )
      #if li == 1:
      #  pool = pooling.VoxelPooling(gate.irreps_out, self.lmax, [0.5, 0.5, 0.5])#, start=[-2, -2, -2],end=[2, 2, 2])
      #else:
      #  pool = pooling.VoxelPooling(gate.irreps_out, self.lmax, [0.1, 0.1, 0.5])#, start=[-2, -2, -2],end=[2, 2, 2])
      #pool = pooling.EdgePooling(gate.irreps_out, self.lmax)
      pool = pooling.TopKPooling(gate.irreps_out, self.lmax)
      #pool = None
      self.layers.append(e3nn_utils.GraphConvBlock(conv, gate, pool))
      irreps_node = gate.irreps_out
      self.irreps_node_sequence.append(irreps_node)

    irreps_node_output = irreps_node_sequence[-1]
    self.layers.append(
      e3nn_utils.GraphConvBlock(
        GraphConvolution(
          irreps_node,
          self.irreps_edge_attrs[-1],
          irreps_node_output,
          fc_neurons,
        )
      )
    )
    self.irreps_node_sequence.append(irreps_node_output)

    self.irreps_node_input = self.irreps_node_sequence[0]
    self.irreps_node_output = self.irreps_node_sequence[-1]

  def forward(self, data) -> torch.Tensor:
    for i, lay in enumerate(self.layers):
      # Edge attributes
      edge_sh = o3.spherical_harmonics(
        range(self.lmax[i] + 1), data.edge_vec, True, normalization="component"
      )
      data.edge_attr_sh = edge_sh

      # Edge length embedding
      edge_length = data.edge_vec.norm(dim=1)
      data.edge_scalars = soft_one_hot_linspace(
        edge_length,
        0.0,
        self.max_radius,
        self.number_of_basis,
        basis="smooth_finite",  # the smooth_finite basis with cutoff = True goes to zero at max_radius
        cutoff=True,  # no need for an additional smooth cutoff
      ).mul(self.number_of_basis**0.5)

      # Forward
      data = lay(data)

    return data.batch, data.x

@compile_mode("script")
class GraphConvolution(torch.nn.Module):
  def __init__(
    self, irreps_node_input, irreps_edge_attr, irreps_node_output, fc_neurons
  ) -> None:
    super().__init__()
    self.irreps_node_input = o3.Irreps(irreps_node_input)
    self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
    self.irreps_node_output = o3.Irreps(irreps_node_output)
    self.sc = o3.Linear(self.irreps_node_input, self.irreps_node_output)
    # self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

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

    assert irreps_mid.dim > 0, (
      f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing "
      f"in irreps_node_output={self.irreps_node_output}"
    )
    instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

    tp = o3.TensorProduct(
      self.irreps_node_input,
      self.irreps_edge_attr,
      irreps_mid,
      instructions,
      internal_weights=False,
      shared_weights=False,
    )
    self.fc = enn.FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
    self.tp = tp

    self.lin = o3.Linear(irreps_mid, self.irreps_node_output)
    # self.lin2 = o3.FullyConnectedTensorProduct(...)

    # inspired by https://arxiv.org/pdf/2002.10444.pdf
    self.alpha = o3.Linear(irreps_mid, "0e")
    with torch.no_grad():
      self.alpha.weight.zero_()

  def forward(self, data) -> torch.Tensor:
    weight = self.fc(data.edge_scalars)

    node_self_connection =  self.sc(data.x)
    edge_features = self.tp(data.x[data.edge_index[0]], data.edge_attr_sh, weight)
    node_features = torch_geometric.utils.scatter(edge_features, data.edge_index[1], dim=0, dim_size=data.x.shape[0], reduce='mean')

    alpha = self.alpha(node_features)
    node_conv_out = self.lin(node_features)

    m = self.sc.output_mask
    alpha = (1 - m) + alpha * m
    return node_self_connection + alpha * node_conv_out

