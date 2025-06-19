from typing import Tuple
import math
import torch
import torch.nn as nn
import torch_geometric.utils
from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils.repeat import repeat as torch_geo_repeat

from e3nn import nn as enn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode

from ..dedm import e3nn_utils

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

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter(node_pos[:,0].cpu(), node_pos[:,1].cpu(), node_pos[:,2].cpu())
    #plt.show()

    perm = e3nn_utils.topk(score, self.ratio, data.batch)
    data.x, data.pos = data.x[perm], data.pos[perm]
    data.batch = data.batch[perm]
    data.edge_index = e3nn_utils.filter_adj(data.edge_index, data.edge_attr, perm, num_nodes=score.size(0))
    data.x *= score[perm].view(-1, 1)
    data.edge_vec = self.getEdgeLengths(data.pos, data.edge_index)

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter(node_pos[:,0].cpu(), node_pos[:,1].cpu(), node_pos[:,2].cpu())
    #plt.show()

    return data

@compile_mode("script")
class VoxelPooling(GraphPooling):
  def __init__(self, irreps_in, lmax, voxel_size, start=None, end=None) -> None:
    super().__init__(irreps_in, lmax)
    self.voxel_size = voxel_size
    self.start = start
    self.end = end

  def forward(self, data) -> torch.Tensor:
    pos = torch.cat([data.pos, data.batch.view(-1,1).to(data.pos.dtype)], dim=-1)
    dim = data.pos.size(1)

    voxel_size = torch.tensor(self.voxel_size, dtype=pos.dtype, device=pos.device)
    voxel_size = torch_geo_repeat(voxel_size, dim)
    voxel_size = torch.cat([voxel_size, voxel_size.new_ones(1)]) # Add batch dim

    start = self.start
    if start is not None:
      start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
      start = torch_geo_repeat(start, dim)
      start = torch.cat([start, start.new_ones(1)]) # Add batch dim

    end = self.end
    if end is not None:
      end = torch.tensor(end, dtype=pos.dtype, device=pos.device)
      end = torch_geo_repeat(end, dim)
      end = torch.cat([end, end.new_ones(1)]) # Add batch dim

    clusters = grid_cluster(pos, voxel_size, start, end)

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.axes.set_xlim3d(left=-2, right=2)
    #ax.axes.set_ylim3d(bottom=-2, top=2)
    #ax.axes.set_zlim3d(bottom=-2, top=2)
    #ax.scatter(data.pos[:,0].cpu(), data.pos[:,1].cpu(), data.pos[:,2].cpu())
    #plt.show()

    data = e3nn_utils.avg_pool(clusters, data)
    data.edge_vec = self.getEdgeLengths(data.pos, data.edge_index)

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.axes.set_xlim3d(left=-2, right=2)
    #ax.axes.set_ylim3d(bottom=-2, top=2)
    #ax.axes.set_zlim3d(bottom=-2, top=2)
    #ax.scatter(data.pos[:,0].cpu(), data.pos[:,1].cpu(), data.pos[:,2].cpu())
    #plt.show()

    return data

@compile_mode("script")
class EdgePooling(GraphPooling):
  def __init__(self, irreps_in, lmax) -> None:
    super().__init__(irreps_in, lmax)

    self.lin = o3.Linear(2 * irreps_in, o3.Irreps("0e"))

  def forward(self, data) -> torch.Tensor:
    e = torch.cat([data.x[data.edge_index[0]], data.x[data.edge_index[1]]], dim=-1)
    e = self.lin(e).view(-1)
    e = torch_geometric.utils.softmax(e, data.edge_index[1], num_nodes=data.x.size(0))

    #data.x, data.pos, data.node_attr, data.edge_index, data.edge_attr, data.batch = self.mergeEdges(data, e)
    data.x, data.pos, data.edge_index, data.batch = self.mergeEdges(data, e)
    data.edge_vec = self.getEdgeLengths(data.pos, data.edge_index)

    return data

  def mergeEdges(self, data, edge_score) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cluster = torch.empty_like(data.batch)
    perm: List[int] = torch.argsort(edge_score, descending=True).tolist()

    # Iterate through all edges, selecting it if it is not incident to
    # another already chosen edge.
    mask = torch.ones(data.x.size(0), dtype=torch.bool)

    i = 0
    new_edge_indices: List[int] = []
    edge_index_cpu = data.edge_index.cpu()
    for edge_idx in perm:
      source = int(edge_index_cpu[0, edge_idx])
      if not bool(mask[source]):
        continue

      target = int(edge_index_cpu[1, edge_idx])
      if not bool(mask[target]):
        continue

      new_edge_indices.append(edge_idx)

      cluster[source] = i
      mask[source] = False

      if source != target:
        cluster[target] = i
        mask[target] = False

      i += 1

    # The remaining nodes are simply kept:
    j = int(mask.sum())
    cluster[mask] = torch.arange(i, i + j, device=data.x.device)
    i += j

    # We compute the new features as an addition of the old ones.
    new_x = torch_geometric.utils.scatter(data.x, cluster, dim=0, dim_size=i, reduce='sum')
    new_pos = torch_geometric.utils.scatter(data.pos, cluster, dim=0, dim_size=i, reduce='mean')
    new_edge_score = edge_score[new_edge_indices]
    if int(mask.sum()) > 0:
      remaining_score = data.x.new_ones(
        (new_x.size(0) - len(new_edge_indices), ))
      new_edge_score = torch.cat([new_edge_score, remaining_score])
    new_x = new_x * new_edge_score.view(-1, 1)

    new_edge_index = torch_geometric.utils.coalesce(cluster[data.edge_index], num_nodes=new_x.size(0))
    new_batch = data.x.new_empty(new_x.size(0), dtype=torch.long)
    new_batch = new_batch.scatter_(0, cluster, data.batch)

    #new_node_attr = torch.ones(new_x.size(0), 1, device=data.x.device)
    #new_edge_attr = torch.ones(new_edge_index.size(1), 1, device=data.x.device)

    #return new_x, new_pos, new_node_attr, new_edge_index, new_edge_attr, new_batch
    return new_x, new_pos, new_edge_index, new_batch
