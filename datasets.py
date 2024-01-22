import pandas as pd
import math
import torch
import numpy as np
import os
from hydra import utils

class DragDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, norm_features, DEVICE='cpu'):
        
        df = pd.read_csv(utils.to_absolute_path(file_name), delim_whitespace=True, header=None)
        df = df[:-1]
        
        in_vars = df.iloc[:, :5].to_numpy()
        # minmax normalization of non-orientation features
        if norm_features:
            in_vars = (in_vars-in_vars.min()) / (in_vars.max() - in_vars.min())
        
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