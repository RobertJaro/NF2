import numpy as np
import torch
from torch import nn

from nf2.data.util import cartesian_to_spherical, vector_cartesian_to_spherical


class ForceFreeLoss(nn.Module):

    def __init__(self, stretch=False):
        super().__init__()
        self.stretch = stretch

    def forward(self, b, jac_matrix, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dx = jac_matrix[:, 1, 0]
        dBz_dx = jac_matrix[:, 2, 0]
        dBx_dy = jac_matrix[:, 0, 1]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dy = jac_matrix[:, 2, 1]
        dBx_dz = jac_matrix[:, 0, 2]
        dBy_dz = jac_matrix[:, 1, 2]
        dBz_dz = jac_matrix[:, 2, 2]
        #
        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        #
        j = torch.stack([rot_x, rot_y, rot_z], -1)
        jxb = torch.cross(j, b, -1)
        normalization = (torch.sum(b ** 2, dim=-1) + 1e-7)
        force_free_loss = torch.sum(jxb ** 2, dim=-1) / normalization
        #
        if self.stretch:
            force_free_loss = torch.asinh(force_free_loss)
        #
        return force_free_loss.mean()


class DivergenceLoss(nn.Module):

    def forward(self, jac_matrix, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dz = jac_matrix[:, 2, 2]

        divergence_loss = (dBx_dx + dBy_dy + dBz_dz) ** 2

        return divergence_loss.mean()

class RadialLoss(nn.Module):

    def __init__(self, base_radius=2.0):
        super().__init__()
        self.base_radius = base_radius

    def forward(self, random_b, random_coords,  *args, **kwargs):
        # radial regularization --> vanishing phi and theta components
        radial_regularization = torch.norm(torch.cross(random_b, random_coords, dim=-1), dim=-1)

        radius_weight = torch.sqrt(torch.sum(random_coords ** 2, dim=-1) + 1e-7)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0)

        normalization = torch.norm(random_b, dim=-1) * torch.norm(random_coords, dim=-1) + 1e-7
        radial_regularization = radial_regularization / normalization

        radial_regularization = (radial_regularization * radius_weight).mean()

        return radial_regularization

class PotentialLoss(nn.Module):

    def __init__(self, base_radius=1.3):
        super().__init__()
        self.base_radius = base_radius

    def forward(self, jac_matrix, coords, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dx = jac_matrix[:, 1, 0]
        dBz_dx = jac_matrix[:, 2, 0]
        dBx_dy = jac_matrix[:, 0, 1]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dy = jac_matrix[:, 2, 1]
        dBx_dz = jac_matrix[:, 0, 2]
        dBy_dz = jac_matrix[:, 1, 2]
        dBz_dz = jac_matrix[:, 2, 2]
        #
        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        #
        j = torch.stack([rot_x, rot_y, rot_z], -1)
        potential_loss = torch.sum(j ** 2, dim=-1)

        if self.base_radius:
            radius_weight = torch.sqrt(torch.sum(coords ** 2, dim=-1) + 1e-7)
            radius_weight = torch.clip(radius_weight - self.base_radius, min=0)
            potential_loss *= radius_weight

        return potential_loss.mean()

class EnergyGradientLoss(nn.Module):

    def __init__(self, base_radius=1.3):
        super().__init__()
        self.base_radius = base_radius
        # self.asinh_stretch = nn.Parameter(torch.tensor(np.arcsinh(1e3), dtype=torch.float32), requires_grad=False)

    def forward(self, b, jac_matrix, coords, n_boundary_coords=None, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dx = jac_matrix[:, 1, 0]
        dBz_dx = jac_matrix[:, 2, 0]
        dBx_dy = jac_matrix[:, 0, 1]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dy = jac_matrix[:, 2, 1]
        dBx_dz = jac_matrix[:, 0, 2]
        dBy_dz = jac_matrix[:, 1, 2]
        dBz_dz = jac_matrix[:, 2, 2]
        # E = b^2 = b_x^2 + b_y^2 + b_z^2
        # dE/dx = 2 * (b_x * dBx_dx + b_y * dBy_dx + b_z * dBz_dx)
        # dE/dy = 2 * (b_x * dBx_dy + b_y * dBy_dy + b_z * dBz_dy)
        # dE/dz = 2 * (b_x * dBx_dz + b_y * dBy_dz + b_z * dBz_dz)
        dE_dx = 2 * (b[:, 0] * dBx_dx + b[:, 1] * dBy_dx + b[:, 2] * dBz_dx)
        dE_dy = 2 * (b[:, 0] * dBx_dy + b[:, 1] * dBy_dy + b[:, 2] * dBz_dy)
        dE_dz = 2 * (b[:, 0] * dBx_dz + b[:, 1] * dBy_dz + b[:, 2] * dBz_dz)

        coords_spherical = cartesian_to_spherical(coords, f=torch)
        t = coords_spherical[:, 1]
        p = coords_spherical[:, 2]
        dE_dr = (torch.sin(t) * torch.cos(p)) * dE_dx + \
                (torch.sin(t) * torch.sin(p)) * dE_dy + \
                torch.cos(p) * dE_dz

        radius_weight = coords[n_boundary_coords:].pow(2).sum(-1).pow(0.5)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0)

        sampled_dE_dr = dE_dr[n_boundary_coords:]
        energy_gradient_regularization = torch.relu(sampled_dE_dr) * radius_weight ** 2
        # energy_gradient_regularization = torch.asinh(energy_gradient_regularization * 1e3) / self.asinh_stretch
        energy_gradient_regularization = energy_gradient_regularization.mean()

        return energy_gradient_regularization

class NaNLoss(nn.Module):

    def forward(self, boundary_b, b_true):
        if torch.isnan(b_true).sum() == 0:
            min_energy_NaNs_regularization = torch.zeros((1,), device=b_true.device)
        else:
            min_energy_NaNs_regularization = boundary_b[torch.isnan(b_true)].pow(2).sum() / torch.isnan(b_true).sum()
        return min_energy_NaNs_regularization

class AzimuthBoundaryLoss(nn.Module):

    def __init__(self, disambiguate=True):
        super().__init__()
        self.disambiguate = disambiguate

    def forward(self, boundary_b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        b_pred = torch.einsum('ijk,ik->ij', transform, boundary_b) if transform is not None else boundary_b
        b_pred = img_to_los_trv_azi(b_pred)

        bz_true = b_true[:, 0]

        if self.disambiguate:
            bx_true = b_true[:, 1] * torch.abs(torch.sin(b_true[:, 2]))
            by_true = b_true[:, 1] * torch.abs(torch.cos(b_true[:, 2]))
        else:
            bx_true = b_true[:, 1] * torch.sin(b_true[:, 2])
            by_true = b_true[:, 1] * torch.cos(b_true[:, 2])
        b_true = torch.stack([bx_true, by_true, bz_true], -1)

        bz = b_pred[:, 0]
        if self.disambiguate:
            bx = b_pred[:, 1] * torch.abs(torch.sin(b_pred[:, 2]))
            by = b_pred[:, 1] * torch.abs(torch.cos(b_pred[:, 2]))
        else:
            bx = b_pred[:, 1] * torch.sin(b_pred[:, 2])
            by = b_pred[:, 1] * torch.cos(b_pred[:, 2])
        b_pred = torch.stack([bx, by, bz], -1)

        # compute diff
        b_diff = torch.abs(b_pred - b_true)
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        return b_diff

def img_to_los_trv_azi(transformed_b):
    eps = 1e-7
    fld = transformed_b.pow(2).sum(-1).pow(0.5)
    cos_inc = transformed_b[..., 2] / (fld + eps)
    azi = torch.arctan2(transformed_b[..., 0], -transformed_b[..., 1] + eps)
    B_los = fld * cos_inc
    B_trv = fld * (1 - cos_inc ** 2) ** 0.5
    transformed_b = torch.stack([B_los, B_trv, azi], -1)
    return transformed_b


class BoundaryLoss(nn.Module):

    def forward(self, boundary_b, b_true, transform=None, b_err=None, *args, **kwargs):
        # apply transforms
        transformed_b = torch.einsum('ijk,ik->ij', transform, boundary_b) if transform is not None else boundary_b
        # compute diff
        b_diff = torch.clip(torch.abs(transformed_b - b_true) - b_err, 0)
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        assert not torch.isnan(b_diff), 'b_diff is nan'
        return b_diff

class HeightLoss(nn.Module):

    def forward(self, boundary_coords, original_coords, height_ranges, *args, **kwargs):
        height_diff = torch.abs(boundary_coords[:, 2] - original_coords[:, 2])
        normalization = (height_ranges[:, 1] - height_ranges[:, 0]) + 1e-6
        height_regularization = torch.true_divide(height_diff, normalization).mean()

        return height_regularization

class SphericalTransform(nn.Module):

    def forward(self, b, b_true, transform=None):
        b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        return b, b_true


class AzimuthTransform(nn.Module):

    def forward(self, b, b_true, transform=None):
        b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        b = img_to_los_trv_azi(b)

        # bz_true = b_true[:, 0]
        # bx_true = b_true[:, 1] * torch.sin(b_true[:, 2])
        # by_true = b_true[:, 1] * torch.cos(b_true[:, 2])
        # b_true = torch.stack([bx_true, by_true, bz_true], -1)
        #
        # bz = b[:, 0]
        # bx = b[:, 1] * torch.sin(b[:, 2])
        # by = b[:, 1] * torch.cos(b[:, 2])
        # b = torch.stack([bx, by, bz], -1)

        return b, b_true

class FluxPreservationLoss(nn.Module):

    def forward(self, dflux_dr, *args, **kwargs):
        flux_preservation_loss = dflux_dr.pow(2).mean()
        return flux_preservation_loss