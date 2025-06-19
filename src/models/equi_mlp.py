import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn.nn as esnn
import escnn
from escnn import gspaces

# src/models/equi_mlp.py

class GSpaceInfo:
    """
    Contains input representations for possible satellite symmetry groups
    """
    def __init__(self, group_act: str):
        if group_act == "c2c2c2":
            gd2 = escnn.group.cyclic_group(2)
            self.group = gspaces.no_base_space(
                escnn.group.direct_product(escnn.group.direct_product(gd2, gd2), gd2)
            )
            self.input_reps = [
                escnn.group.directsum(
                    [
                        self.group.irrep(((0,), (1,), 0), (0,)),
                        self.group.irrep(((0,), (0,), 0), (1,)),
                        self.group.irrep(((1,), (0,), 0), (0,)),
                    ]
                )
            ]
        elif group_act == "c2c2":
            gd2 = escnn.group.cyclic_group(2)
            self.group = gspaces.no_base_space(escnn.group.direct_product(gd2, gd2))
            self.input_reps = 1 * [
                self.group.irrep((0,), (1,), 0) + self.group.irrep((1,), (0,), 0)
            ] + 1 * [self.group.trivial_repr]
        elif group_act == "c2":
            self.group = gspaces.no_base_space(escnn.group.cyclic_group(2))
            self.input_reps = (
                1 * [self.group.trivial_repr]
                + 1 * [self.group.irrep(1)]
                + 1 * [self.group.trivial_repr]
            )
        elif group_act == "trivial":
            self.group = gspaces.no_base_space(escnn.group.cyclic_group(1))
            self.input_reps = 3 * [self.group.trivial_repr]
        elif group_act == "c4":
            self.group = escnn.group.cyclic_group(4)
            raise NotImplementedError()
        elif group_act == "oh":
            self.group = gspaces.no_base_space(escnn.group.octa_group())
            self.input_reps = 1 * [self.group.fibergroup.standard_representation]
        else:
            raise NotImplementedError(f"Group {group_act} not found")


class InvariantDragMLP(nn.Module):
    def __init__(
        self,
        gspaceinfo: GSpaceInfo,
        hidden_dim=128,
        norm_features=False,
        norm_min=None,
        norm_max=None,
    ):
        act = gspaceinfo.group
        self.act = act
        input_reps = gspaceinfo.input_reps
        self.input_reps = input_reps

        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_features = norm_features

        self.norm_min = norm_min
        self.norm_max = norm_max

        self.network = esnn.SequentialModule(
            esnn.Linear(
                esnn.FieldType(act, 1 * input_reps),
                esnn.FieldType(act, hidden_dim * [act.regular_repr]),
            ),
            esnn.ReLU(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.Linear(
                esnn.FieldType(act, hidden_dim * [act.regular_repr]),
                esnn.FieldType(act, hidden_dim * [act.regular_repr]),
            ),
            esnn.ReLU(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.GroupPooling(esnn.FieldType(act, hidden_dim * [act.regular_repr])),
            esnn.Linear(
                esnn.FieldType(act, hidden_dim * [act.trivial_repr]),
                esnn.FieldType(act, hidden_dim * [act.trivial_repr]),
            ),
        )

        self.network2 = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        dir_vec = x[:, 0:3].reshape((x.shape[0], 3))  # orientation
        in_vars = x[:, 3:]  # temperature, velocity, etc.
        if self.norm_features:
            in_vars = (in_vars - self.norm_min) / (self.norm_max - self.norm_min)

        geo_x = self.network.in_type(dir_vec)
        h = self.network(geo_x).tensor.reshape(x.shape[0], self.hidden_dim)
        cat = torch.cat((h, in_vars), axis=1)
        h2 = self.network2(cat)

        return h2
