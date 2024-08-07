import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn.nn as enn
import escnn
from escnn import gspaces
from groups import GSpaceInfo

class InvariantDragMLP(nn.Module):
    def __init__(self, gspaceinfo: GSpaceInfo, hidden_dim=128, norm_features=False, norm_min=None, norm_max=None):
        act = gspaceinfo.group
        input_reps = gspaceinfo.input_reps

        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_features = norm_features

        self.norm_min = norm_min
        self.norm_max = norm_max

        self.network = enn.SequentialModule(
             
            enn.Linear(enn.FieldType(act, 1 * input_reps),
                       enn.FieldType(act, hidden_dim * [act.regular_repr])),
            enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr])),
            enn.Linear(enn.FieldType(act, hidden_dim * [act.regular_repr]),
                       enn.FieldType(act, hidden_dim * [act.regular_repr])),
            enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr])),
            enn.GroupPooling(enn.FieldType(act, hidden_dim * [act.regular_repr])),
            enn.Linear(enn.FieldType(act, hidden_dim * [act.trivial_repr]),
                       enn.FieldType(act, hidden_dim * [act.trivial_repr])),
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
