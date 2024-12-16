import pandas as pd
import math
import torch
import numpy as np
import os
from hydra import utils
import trimesh
from torch_geometric.data import Data
from e3nn import o3

class DragDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, norm_features, DEVICE='cpu'):
        
        df = pd.read_csv(utils.to_absolute_path(file_name), delim_whitespace=True, header=None)
        df = df[:-1]
        
        in_vars = df.iloc[:, :5].to_numpy()
        # minmax normalization of non-orientation features
        if norm_features:
            in_vars = (in_vars-in_vars.min(axis=0)) / (in_vars.max(axis=0) - in_vars.min(axis=0))
        
        orientation = df.iloc[:, 5:7].values
        y = df.iloc[:, 7].values
        
        dir_vec = []
        for yaw_pitch in orientation:
            dir_x = math.cos(yaw_pitch[0]) * math.cos(yaw_pitch[1])
            dir_y = math.sin(yaw_pitch[1])
            dir_z = math.sin(yaw_pitch[0]) * math.cos(yaw_pitch[1])
            dir_vec.append([dir_y, dir_x, dir_z])
        
        cat = np.concatenate((np.array(dir_vec), in_vars), axis=1)
        
        self.x = torch.tensor(cat, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

# we are just overfitting to one mesh for now because of lack of mesh variety (but still equivariant)
class DragMeshDataset(torch.utils.data.Dataset):
    def __init__(self, attr_file, mesh_file, norm_features=True, DEVICE='cpu', data_lim=-1, return_features_separately=False):
        super().__init__()
        self.device = DEVICE
        mesh = trimesh.load(mesh_file, file_type="stl", force="mesh")
        self.vertices_canonical = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype()).to(DEVICE)
        self.edges = torch.tensor(mesh.edges_unique).to(DEVICE)
        self.return_features_separately = return_features_separately

        df = pd.read_csv(utils.to_absolute_path(attr_file), delim_whitespace=True, header=None)
        df = df[:-1]

        if data_lim is not -1:
            df = df[:data_lim]
        
        attrs = df.iloc[:, :5].to_numpy()
        if norm_features:
            attrs = (attrs-attrs.min(axis=0)) / (attrs.max(axis=0) - attrs.min(axis=0))
        self.xs = torch.tensor(attrs, dtype=torch.float32).to(DEVICE)
        
        # needs to be CPU for D_from_angles
        self.orientation = torch.tensor(df.iloc[:, 5:7].values, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(df.iloc[:, 7].values, dtype=torch.float32).to(DEVICE)

    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # calculate edge_vec
        edge_vec = self.vertices_canonical[self.edges[:, 1]] - self.vertices_canonical[self.edges[:, 0]]
        if self.return_features_separately:
            return Data(
                pos=self.vertices_canonical,
                x=torch.ones_like(self.xs[idx]).repeat(self.vertices_canonical.size(0), 1),
                edge_index=self.edges.T,
                edge_vec=edge_vec,
                orientation=self.orientation[idx].unsqueeze(0),
                feats=self.xs[idx]
            ), self.y[idx]
        else:
            return Data(
                pos=self.vertices_canonical,
                x=self.xs[idx].repeat(self.vertices_canonical.size(0), 1),
                edge_index=self.edges.T,
                edge_vec=edge_vec,
                orientation=self.orientation[idx].unsqueeze(0),
            ), self.y[idx]

    def _rotate(self, data, orientation):
        vertices = data.pos
        vertices_rot = vertices @ o3.Irrep("1e").D_from_angles(orientation[0], orientation[1], torch.tensor(0))
        data_rot = data.clone()
        data_rot.pos = vertices_rot
        return data_rot
    
# if __name__ == "__main__":
#     df = pd.read_csv(utils.to_absolute_path("data/cube50k.dat"), delim_whitespace=True, header=None)
#     df = df[:-1]
        
#     in_vars = df.iloc[:, :5].to_numpy()
#     # minmax normalization of non-orientation features
#     # apply to each column separately
#     in_vars = (in_vars-in_vars.min(axis=0)) / (in_vars.max(axis=0) - in_vars.min(axis=0))


if __name__ == "__main__":
    
    ds = DragMeshDataset("data/cube50k.dat", "STLs/Cube_38_1m.stl")
    print(ds[2])
