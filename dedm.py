import rem.e3nn_utils as e3nn_utils
from typing import Dict

import torch
import torch as tr
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import SO3Activation

from gate_points_message_passing import MessagePassing


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


def radius_graph(pos, r_max, batch) -> torch.Tensor:
    # naive and inefficient version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > 0)).nonzero().T
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index

def radius_graph_(pos, r_max) -> torch.Tensor:
    # naive version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    return ((r < r_max) & (r > 0)).nonzero().T


class SimpleNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=50,
        encoder_layers=3,
        decoder_layers=1,
        decoder_layer_hiddens=[64],
        lmax=2,
        f_out=16,
        pool_nodes=True,
        rotate="pos"
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        # so3 irreps
        irreps_enc_out = o3.Irreps(
            [(f_out, (l, p)) for l in range((lmax) + 1) for p in [-1,1]]
        )
        grid_so3 = e3nn_utils.so3_near_identity_grid()
        grid_s2 = e3nn_utils.s2_near_identity_grid()

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_in] + encoder_layers * [irreps_node_hidden] + [irreps_enc_out],
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_latent = e3nn_utils.so3_irreps(lmax)
        self.irreps_enc_out = self.mp.irreps_node_output

        self.enc_to_so3 = o3.Linear(self.irreps_enc_out, self.irreps_latent, f_in=1, f_out=f_out)

        if decoder_layers == 0:
            decoder_layer_hiddens.append(f_out)
        
        self.decoder_layers_list = torch.nn.ModuleList()
        for i in range(decoder_layers):
            if i == 0:
                self.decoder_layers_list.append(
                    e3nn_utils.SO3Convolution(f_out, decoder_layer_hiddens[i], lmax, kernel_grid=grid_so3)
                )
            else:
                self.decoder_layers_list.append(
                    e3nn_utils.SO3Convolution(decoder_layer_hiddens[i - 1], decoder_layer_hiddens[i], lmax, kernel_grid=grid_so3)
                )
            self.decoder_layers_list.append(
                SO3Activation(self.lmax, self.lmax, torch.relu, resolution=12)
            )

        self.lin = e3nn_utils.SO3ToS2Convolution(
            decoder_layer_hiddens[-1], 1, lmax, kernel_grid=grid_s2
        )
        # s2 irreps
        self.irreps_out = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

        # handles if we rotate the positions of the satellite according to the orientation, or
        # if we query the predicted spherical harmonic signal at (alpha, beta)
        if rotate not in ["pos", "query"]:
            raise ValueError(f"Invalid value for rotate: {rotate}")
        
        self.rotate = rotate

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = data.batch.max() + 1
        if self.rotate == "pos":
            self._rotate_positions(data)
        
        # compute edge attributes for rotated data
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        batch_size = data.batch.max() + 1

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

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

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            enc_out = scatter(node_outputs, batch, int(batch.max()) + 1).div(self.num_nodes**0.5)
        else:
            enc_out = node_outputs

        enc_out = self.enc_to_so3(enc_out.view(batch_size, 1, -1))
        
        for layer in self.decoder_layers_list:
            enc_out = layer(enc_out)
        
        enc_out = self.lin(enc_out)
        print(enc_out)

        if self.rotate == "query":    
            cartesian = self._ar2los(data.orientation)
        else:
            cartesian = torch.tensor([[0., 0., 1]])
        
        out_response = self._getResponse(enc_out, cartesian, batch_size)
        
        return out_response
    
    def _getResponse(self, out, pose, batch_size):
        sh = torch.concatenate(
            [o3.spherical_harmonics(l, pose, True) for l in range(self.lmax + 1)], dim=1
        ).unsqueeze(2)  # B x (L^2 * S^2) x 1
        response = torch.bmm(out, sh).squeeze()  # B x D

        return response
    
    def _ar2los(self, x_ar):
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
    
    def _rotate_positions(self, data: Dict[str, torch.Tensor]) -> None:
        # Step 1. Gather alpha/beta per node (by indexing the batch's alpha/beta)
        alpha = data.orientation[data.batch, 0]  # shape [N]
        beta  = data.orientation[data.batch, 1]  # shape [N]

        # Step 2. Build R_alpha, R_beta, R per node
        cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
        cos_beta,  sin_beta  = torch.cos(beta),  torch.sin(beta)

        R_alpha = torch.stack(
            [
            cos_alpha, -sin_alpha, torch.zeros_like(alpha),
            sin_alpha,  cos_alpha, torch.zeros_like(alpha),
            torch.zeros_like(alpha), torch.zeros_like(alpha), torch.ones_like(alpha)
            ],
            dim=-1
        ).view(-1, 3, 3)  # shape [N, 3, 3]

        R_beta = torch.stack(
            [
            cos_beta, torch.zeros_like(beta),  sin_beta,
            torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
            -sin_beta, torch.zeros_like(beta), cos_beta
            ],
            dim=-1
        ).view(-1, 3, 3)  # shape [N, 3, 3]

        R = torch.bmm(R_beta, R_alpha)  # shape [N, 3, 3]

        # Step 3. Rotate each position
        # data.pos is [N, 3], so unsqueeze -> [N, 1, 3],
        # bmm with [N, 3, 3] -> [N, 1, 3], then squeeze -> [N, 3].
        data.pos = torch.bmm(data.pos.unsqueeze(1), R).squeeze(1)


if __name__ == "__main__":
    from datasets import DragMeshDataset
    from torch_geometric.data import DataLoader
    import random

    ds = DragMeshDataset("/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/data/cube50k.dat", 
                     "/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/STLs/Cube_38_1m.stl", 
                     return_features_separately=False)
    dl = iter(DataLoader(ds, batch_size=1, shuffle=False))
    
    irreps_in = "5x0e"
    lmax = 3
    f_out = 2

    net = SimpleNetwork(irreps_in, encoder_layers=1, decoder_layers=1, decoder_layer_hiddens=[64], max_radius=1.7, num_neighbors=3.0, num_nodes=5.0, 
                        f_out=f_out, lmax=lmax, rotate="pos").eval()
    print(net.mp.irreps_node_sequence)
    print(net.irreps_latent)
    print(net.irreps_out)
    print()
    
    samp, y = next(dl)
    
    samp_copy = samp.clone()
    
    # equivariance error
    cd = net(samp)
    print(cd)
    exit(0)
    
    orientations = [
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[torch.pi / 2, 0.0]]),
        torch.tensor([[0.0, torch.pi / 2]]),
        torch.tensor([[torch.pi / 2, torch.pi / 2]]),
        torch.tensor([[torch.pi, 0.0]]),
        torch.tensor([[0.0, torch.pi]]),
        torch.tensor([[torch.pi, torch.pi]]),
    ]

    print("Following should be invariant: ")

    for orientation in orientations:
        samp_copy.orientation[:] = orientation
        cd_varied = net(samp_copy)
        print(f"Orientation: {orientation}")
        print(f"cd: {cd_varied}")
        print()

    print("Following should be random: ")
    
    # Generate random orientations
    num_random_orientations = 20
    random_orientations = [
        torch.tensor([[random.uniform(0, 2 * torch.pi), random.uniform(0, 2 * torch.pi)]]) 
        for _ in range(num_random_orientations)
    ]

    # Evaluate the network on the random orientations
    for orientation in random_orientations:
        samp_copy.orientation[:] = orientation  # Update orientation
        cd_varied = net(samp_copy)  # Pass through the network
        print(f"Orientation: {orientation}")
        print(f"cd: {cd_varied}")
        print()
