from typing import Dict

import torch
import torch_geometric.utils
from e3nn import o3
from e3nn import nn as enn
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode
import math
from torch_geometric.nn.pool.consecutive import consecutive_cluster


class AttrGNN(torch.nn.Module):
  def __init__(
    self,
    irreps_node_input,
    irreps_node_attr,
    irreps_node_output,
    max_radius,
    mul=50,
    layers=3,
    lmax=2,
    pool_nodes=True,
  ) -> None:
    super().__init__()

    self.lmax = [lmax, lmax, lmax]
    self.max_radius = max_radius
    self.number_of_basis = 10
    self.pool_nodes = pool_nodes

    irreps_node_hiddens = list()
    irreps_edge_hiddens = list()
    for layer_lmax in self.lmax[:-1]:
      irreps_node_hiddens.append(o3.Irreps(
        [(mul, (l, p)) for l in range(layer_lmax + 1) for p in [-1, 1]]
      ))
      irreps_edge_hiddens.append(o3.Irreps.spherical_harmonics(layer_lmax))
    

    irreps_node_seq = [irreps_node_input] + irreps_node_hiddens + [irreps_node_output]
    irreps_edge_seq = irreps_edge_hiddens + [o3.Irreps.spherical_harmonics(self.lmax[-1])]
    self.mp = AttrMessagePassing(
      irreps_node_sequence=irreps_node_seq,
      irreps_node_attr=irreps_node_attr,
      irreps_edge_attrs=irreps_edge_seq,
      fc_neurons=[self.number_of_basis, 100],
      max_radius=max_radius,
      lmax=self.lmax
    )
    self.irreps_node_input = self.mp.irreps_node_input
    self.irreps_node_attr = self.mp.irreps_node_attr
    self.irreps_node_output = self.mp.irreps_node_output

  def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "batch" in data:
      batch = data["batch"]
    else:
      batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

    # Create graph
    if "edge_index" in data:
      edge_src = data["edge_index"][0]
      edge_dst = data["edge_index"][1]
    else:
      edge_index = radius_graph(data["pos"], self.max_radius, batch)
      edge_src = edge_index[0]
      edge_dst = edge_index[1]

    # Edge attributes
    edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

    if "x" in data:
        node_input = data["x"]
    else:
        node_input = data["node_input"]

    node_attr = data["node_attr"]
    edge_attr = data["edge_attr"]

    return batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec


  def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec = self.preprocess(data)
    del data

    # Edge attributes
    edge_sh = o3.spherical_harmonics(range(self.lmax[-1] + 1), edge_vec, True, normalization="component")
    edge_attr = torch.cat([edge_attr, edge_sh], dim=1)

    # Edge length embedding
    edge_length = edge_vec.norm(dim=1)
    edge_length_embedding = soft_one_hot_linspace(
      edge_length,
      0.0,
      self.max_radius,
      self.number_of_basis,
      basis="smooth_finite",  # the smooth_finite basis with cutoff = True goes to zero at max_radius
      cutoff=True,  # no need for an additional smooth cutoff
    ).mul(self.number_of_basis**0.5)

    node_outputs = self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

    if self.pool_nodes:
      return torch_geometric.utils.scatter(node_outputs, batch, dim=0, reduce='mean')
    else:
      return node_outputs

class AttrMessagePassing(torch.nn.Module):
  def __init__(
    self,
    irreps_node_sequence,
    irreps_node_attr,
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
    self.irreps_node_attr = o3.Irreps(irreps_node_attr)
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
          and tp_path_exists(
            irreps_node, irreps_edge_attr, ir
          )
        ]
      ).simplify()
      irreps_gated = o3.Irreps(
        [
          (mul, ir)
          for mul, ir in irreps_node_hidden
          if ir.l > 0
          and tp_path_exists(
            irreps_node, irreps_edge_attr, ir
          )
        ]
      )
      if irreps_gated.dim > 0:
        if tp_path_exists(irreps_node, irreps_edge_attr, "0e"):
          ir = "0e"
        elif tp_path_exists(
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
      conv = GraphConvolutionWithNodeAttrs(
        irreps_node,
        self.irreps_node_attr,
        irreps_edge_attr,
        gate.irreps_in,
        fc_neurons,
      )
      
      pool = TopKPooling(gate.irreps_out, self.lmax)
      self.layers.append(GraphConvBlock(conv, gate, pool))
      irreps_node = gate.irreps_out
      self.irreps_node_sequence.append(irreps_node)

    irreps_node_output = irreps_node_sequence[-1]
    self.layers.append(
      GraphConvBlock(
        GraphConvolutionWithNodeAttrs(
          irreps_node,
          self.irreps_node_attr,
          self.irreps_edge_attrs[-1],
          irreps_node_output,
          fc_neurons,
        )
      )
    )
    self.irreps_node_sequence.append(irreps_node_output)

    self.irreps_node_input = self.irreps_node_sequence[0]
    self.irreps_node_output = self.irreps_node_sequence[-1]

  def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
    for i, lay in enumerate(self.layers):
      # Forward
      data = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

    return data.batch, data.x

@compile_mode("script")
class GraphConvolutionWithNodeAttrs(torch.nn.Module):
  def __init__(
    self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons
  ) -> None:
    super().__init__()
    self.irreps_node_input = o3.Irreps(irreps_node_input)
    self.irreps_node_attr = o3.Irreps(irreps_node_attr)
    self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
    self.irreps_node_output = o3.Irreps(irreps_node_output)

    self.sc = o3.FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)
    self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

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

    self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)

    # inspired by https://arxiv.org/pdf/2002.10444.pdf
    self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
    with torch.no_grad():
      self.alpha.weight.zero_()
    assert (
       self.alpha.output_mask[0] == 1.0
    ), f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

  def forward(self, data) -> torch.Tensor:
    weight = self.fc(data.edge_scalars)

    node_self_connection = self.sc(data.x, data.node_attr)
    node_features = self.lin1(data.x, data.node_attr)

    edge_features = self.tp(node_features[data.edge_index[0]], data.edge_attr_sh, weight)
    node_features = torch_geometric.utils.scatter(edge_features, data.edge_index[1], dim=0, dim_size=data.x.shape[0], reduce='mean')

    alpha = self.alpha(node_features, data.node_attr)
    node_conv_out = self.lin2(node_features, data.node_attr)

    m = self.sc.output_mask
    alpha = (1 - m) + alpha * m
    return node_self_connection + alpha * node_conv_out
  
@compile_mode("script")
class GraphPooling(torch.nn.Module):
  def __init__(self, irreps_in, lmax, num_basis=10, max_radius=5) -> None:
    super().__init__()
    self.irreps_in = irreps_in
    self.lmax = lmax
    self.num_basis = num_basis
    self.max_radius = max_radius

  def getEdgeLengths(self, pos, edge_index) -> torch.Tensor:
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    return edge_vec

@compile_mode("script")
class TopKPooling(GraphPooling):
  def __init__(self, irreps_in, lmax, ratio=0.5, min_score=None, multiplier=1.) -> None:
    super().__init__(irreps_in, lmax)
    self.ratio = ratio
    self.min_score = min_score
    self.multiplier = multiplier

    self.attn = o3.Linear(irreps_in, o3.Irreps("0e"))

  def forward(self, data) -> torch.Tensor:
    score = torch.tanh((data.x * self.attn(data.x)).sum(dim=-1) / self.attn.weight.norm(p=2, dim=-1))

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
  irreps_in1 = o3.Irreps(irreps_in1).simplify()
  irreps_in2 = o3.Irreps(irreps_in2).simplify()
  ir_out = o3.Irrep(ir_out)

  for _, ir1 in irreps_in1:
    for _, ir2 in irreps_in2:
      if ir_out in ir1 * ir2:
        return True
  return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
  # special case of torch_scatter.scatter with dim=0
  out = src.new_zeros(dim_size, src.shape[1])
  index = index.reshape(-1, 1).expand_as(src)
  return out.scatter_add_(0, index, src)

def radius_graph(pos, r_max, batch):
  # naive and inefficient version of torch_cluster.radius_graph
  r = torch.cdist(pos, pos)
  index = ((r < r_max) & (r > 0)).nonzero().T
  index = index[:, batch[index[0]] == batch[index[1]]]
  return index

def avg_pool(cluster, data):
  cluster, perm = consecutive_cluster(cluster)

  data.x = torch_geometric.utils.scatter(data.x, cluster, dim=0, reduce='mean')
  data.pos = torch_geometric.utils.scatter(data.pos, cluster, dim=0, reduce='mean')
  data.node_attr = torch_geometric.utils.scatter(data.node_attr, cluster, dim=0, reduce='mean')
  data.edge_index, data.edge_attr = pool_edge(cluster, data.edge_index, data.edge_attr)
  data.batch = data.batch[perm]

  return data

def topk(x, ratio, batch):
  num_nodes = torch_geometric.utils.scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
  k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

  x, x_perm = torch.sort(x.view(-1), descending=True)
  batch = batch[x_perm]
  batch, batch_perm = torch.sort(batch, descending=False, stable=True)

  arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
  ptr = num_nodes.new_zeros(num_nodes.numel() + 1)
  torch.cumsum(num_nodes, 0, out=ptr[1:])
  batched_arange = arange - ptr[batch]
  mask  = batched_arange < k[batch]

  return x_perm[batch_perm[mask]]

def pool_edge(cluster, edge_index, edge_attr=None):
  num_nodes = cluster.size(0)
  edge_index = cluster[edge_index.view(-1)].view(2, -1)
  edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
  if edge_index.numel() > 0:
    edge_index, edge_attr = torch_geometric.utils.coalesce(edge_index, edge_attr, num_nodes)
  return edge_index, edge_attr

def filter_adj(edge_index, edge_attr, node_index, cluster_index=None, num_nodes=None):
  #num_nodes = max(int(edge_src.max()) + 1, int(edge_dst.max()) + 1)

  if cluster_index is None:
    cluster_index = torch.arange(node_index.size(0), device=node_index.device)

  # Create batch mask
  mask = node_index.new_full((num_nodes, ), -1)
  mask[node_index] = cluster_index

  # Filter out edges that are no longer connected
  row, col = mask[edge_index[0]], mask[edge_index[1]]
  mask = (row >= 0) & (col >= 0)
  row, col = row[mask], col[mask]

  return torch.stack([row, col], dim=0)

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

def so3_near_identity_grid(max_beta=math.pi / 8, max_gamma=2 * math.pi, n_alpha=8, n_beta=3, n_gamma=None):
  """
  :return: rings of rotations around the identity, all points (rotations) in
  a ring are at the same distance from the identity
  size of the kernel = n_alpha * n_beta * n_gamma
  """
  if n_gamma is None:
      n_gamma = n_alpha  # similar to regular representations
  beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
  alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
  pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
  A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
  C = preC - A
  A = A.flatten()
  B = B.flatten()
  C = C.flatten()
  return torch.stack((A, B, C))

# L^2 * S^2 -> natural decomposition for a signal over S^2
def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

# L^2 * SO(3) -> natural decomposition for a signal over SO(3)
def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def flat_wigner(lmax, alpha, beta, gamma):
  return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)

# Model Blocks
class SO3Convolution(nn.Module):
  def __init__(self, f_in, f_out, lmax_in, kernel_grid, lmax_out=None):
    super().__init__()
    if lmax_out == None:
      lmax_out = lmax_in
    self.register_parameter(
      'w', nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    ) # [f_in, f_out, n_so3_pts]
    self.register_parameter(
      'D', nn.Parameter(flat_wigner(lmax_in, *kernel_grid))
    ) # [n_so3_pts, psi]
    self.lin = o3.Linear(so3_irreps(lmax_in), so3_irreps(lmax_out), f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    # S2 DFT (matrix mulitplication)
    psi = torch.einsum('ni,xyn->xyi', self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)

#class SO3ToS2Convolution(nn.Module):
#  def __init__(self, f_in, f_out, lmax, kernel_grid):
#    super().__init__()
#    self.register_parameter(
#      'w', nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
#    ) # [f_in, f_out, n_so3_pts]
#    self.register_parameter(
#      #'D', nn.Parameter(flat_wigner(lmax, *kernel_grid))
#      'D', nn.Parameter(o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization='component'))
#    ) # [n_so3_pts, psi]
#    self.lin = o3.Linear(so3_irreps(lmax), s2_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)
#
#  def forward(self, x):
#    # B x F_in x psi
#    # S2 DFT (matrix mulitplication)
#    psi = torch.einsum('ni,xyn->xyi', self.D, self.w) / self.D.shape[0] ** 0.5
#    return self.lin(x, weight=psi)

class SO3ToS2Convolution(nn.Module):
  def __init__(self, f_in, f_out, lmax_in, lmax_out, kernel_grid):
    super().__init__()
    self.lin = o3.Linear(so3_irreps(lmax_in), s2_irreps(lmax_out), f_in=f_in, f_out=f_out)

  def forward(self, x):
    return self.lin(x)

@compile_mode("script")
class GraphConvBlock(torch.nn.Module):
  def __init__(self, gconv, act=None, pool=None) -> None:
    super().__init__()
    self.gconv = gconv
    self.act = act
    self.pool = pool

  def forward(self, data) -> torch.Tensor:
    node_features = self.gconv(data)

    if self.act is not None:
      node_features = self.act(node_features)
    data.x = node_features

    if self.pool is not None:
      return self.pool(data)
    else: # Return the non-modified graph if not pooling
      return data

if __name__ == "__main__":
  from datasets import DragMeshDataset

  z_lmax = 3
  irreps_node_attr = o3.Irreps("1x0e")
  irreps_in = o3.Irreps("1x0e")
  irreps_latent = so3_irreps(z_lmax)
  irreps_enc_out = o3.Irreps([(38, (l, p)) for l in range((z_lmax) + 1) for p in [-1,1]])
  encoder = AttrGNN(irreps_in, irreps_node_attr, irreps_enc_out, 1.8)
  
  ds = DragMeshDataset("data/cube50k.dat", "STLs/Cube_38_1m.stl")
  encoder(ds[0])
