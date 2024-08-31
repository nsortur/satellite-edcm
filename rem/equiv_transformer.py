import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3
from e3nn.nn import SO3Activation

class TransformerBlock(nn.Module):
  def __init__(self, irreps_in, irreps_query, irreps_key, irreps_out, max_radius):
    self.number_of_basis = 10
    self.max_radius = max_radius

    self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
    self.fc_k = nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel], act=torch.nn.functional.silu)

    self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
    self.fc_v = nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=torch.nn.functional.silu)

    self.irreps_sh = o3.Irreps.spherical_harmonics(3)

  def preprocess(data):
    if "batch" in data:
      batch = data["batch"]
    else:
      batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

    # Create graph
    edge_index = radius_graph(data["pos"], self.max_radius, batch)
    edge_src = edge_index[0]
    edge_dst = edge_index[1]

    # Edge attributes
    edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

    return batch, data["x"], edge_src, edge_dst, edge_vec

  def forward(self, data):
    batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
    del data

    edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

