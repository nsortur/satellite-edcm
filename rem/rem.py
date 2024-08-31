import math
import torch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3
from e3nn.nn import SO3Activation

from . import e3nn_utils
from .equiv_gnn import GNN
from .equiv_gnn_w_attrs import AttrGNN


class Decoder(nn.Module):
  def __init__(self, lmax_in, lmax_out, f_in, f_out, invariant_out=False):
    super().__init__()
    self.invariant_out = invariant_out

    grid_s2 = e3nn_utils.s2_near_identity_grid()
    grid_so3 = e3nn_utils.so3_near_identity_grid()

    self.so3_conv1 = e3nn_utils.SO3Convolution(
      f_in, 64, lmax_in, kernel_grid=grid_so3
    )
    self.act1 = SO3Activation(lmax_in, lmax_out, torch.relu, resolution=12)

    self.so3_conv2 = e3nn_utils.SO3Convolution(
      64, 128, lmax_in, kernel_grid=grid_so3
    )
    self.act2 = SO3Activation(lmax_in, lmax_out, torch.relu, resolution=12)

    self.so3_conv3 = e3nn_utils.SO3Convolution(
      128, 256, lmax_in, kernel_grid=grid_so3
    )

    # Output: Maps to 53 (rho_0, rho_1, rho_2, rho_3, ...) -> 53 S2 signals
    if self.invariant_out:
      self.act3 = SO3Activation(lmax_in, 0, torch.relu, resolution=12)
      self.lin = o3.Linear(256, f_out)
    else:
      self.act3 = SO3Activation(lmax_in, lmax_out, torch.relu, resolution=12)
      self.lin = e3nn_utils.SO3ToS2Convolution(
        256, f_out, lmax_out, kernel_grid=grid_s2
      )

  def forward(self, x):
    x = self.so3_conv1(x)
    x = self.act1(x)

    x = self.so3_conv2(x)
    x = self.act2(x)

    x = self.so3_conv3(x)
    x = self.act3(x)

    x = self.lin(x)

    return x


class REM(nn.Module):
  def __init__(self, num_node_features, z_lmax, max_radius, out_dim, invariant_out=False):
    super().__init__()

#    z_lmax = 4
    self.lmax = z_lmax
    self.out_dim = out_dim
    self.invariant_out = invariant_out
    f = 16

    self.irreps_in = o3.Irreps(f"{num_node_features}x0e")
    self.irreps_latent = e3nn_utils.so3_irreps(z_lmax)
    self.irreps_enc_out = o3.Irreps(
      #[(f, (l, p)) for l in range((z_lmax // 2) + 1) for p in [-1,1]]
      [(f, (l, p)) for l in range((z_lmax) + 1) for p in [-1,1]]
    )
    if self.invariant_out:
      self.irreps_node_attr = o3.Irreps("1x1e")
      self.encoder = AttrGNN(
        irreps_node_input=self.irreps_in,
        irreps_node_attr=self.irreps_node_attr,
        irreps_node_output=self.irreps_enc_out,
        max_radius=max_radius,
        layers=2,
        mul=f,
        lmax=[self.lmax, self.lmax, self.lmax],
      )
    else:
      self.encoder = GNN(
        irreps_node_input=self.irreps_in,
        irreps_node_output=self.irreps_enc_out,
        max_radius=max_radius,
        layers=2,
        mul=f,
        #lmax=[self.lmax // 2, self.lmax // 2, self.lmax // 2],
        lmax=[self.lmax, self.lmax, self.lmax],
      )

    self.lin = o3.Linear(self.irreps_enc_out, self.irreps_latent, f_in=1, f_out=f)
    self.decoder = Decoder(z_lmax, z_lmax, f, out_dim, invariant_out=invariant_out)

  def forward(self, x, return_latent=False):
    batch_size = x.batch.max() + 1
    gnn_out = self.encoder(x)
    z = self.lin(gnn_out.view(batch_size, 1, -1))
    out = self.decoder(z)
    cartesian = self.ar2los(x.orientation)
    out = self._getResponse(out, cartesian)

    if return_latent:
      return out, z
    else:
      return out

  def _getResponse(self, out, pose):
    if self.invariant_out:
      return out
    else:
      sh = torch.concatenate(
        [o3.spherical_harmonics(l, pose, True) for l in range(self.lmax + 1)], dim=1
      ).unsqueeze(2)  # B x (L^2 * S^2) x 1
      response = torch.bmm(out, sh).squeeze()  # B x D

      return response
    
  def ar2los(self, x_ar):
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
