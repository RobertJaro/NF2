import torch
import wandb
from astropy import units as u
from torch import nn

from nf2.data.util import cartesian_to_spherical, img_to_los_trv_azi, los_trv_azi_to_img
from nf2.train.model import jacobian


class BaseLoss(nn.Module):

    def __init__(self, name, ds_id, **kwargs):
        super().__init__()
        self.name = name
        self.ds_id = ds_id


class ForceFreeLoss(BaseLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, b, jac_matrix, coords, *args, **kwargs):
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
        normalization = b.pow(2).sum(-1) + 1e-7
        jxb = torch.cross(j, b, -1)
        force_free_loss = jxb.pow(2).sum(-1) / normalization

        # check for NaNs
        assert not torch.isnan(force_free_loss).any(), 'NaNs in force-free loss computation!'

        return force_free_loss

class SigmaJLoss(BaseLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, b, jac_matrix, coords, *args, **kwargs):
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
        #
        b_norm = torch.norm(b, dim=-1)
        j_norm = torch.norm(j, dim=-1)
        jxb = torch.cross(j, b, -1)

        # current weighted loss of JxB/|J||B|
        loss = torch.norm(jxb, dim=-1) / (b_norm + 1e-7)
        loss = loss.sum() / j_norm.sum()

        return loss


class DivergenceLoss(BaseLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, jac_matrix, coords, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dz = jac_matrix[:, 2, 2]

        divergence_loss = (dBx_dx + dBy_dy + dBz_dz) ** 2

        return divergence_loss


class RadialLoss(BaseLoss):

    def forward(self, b, coords, *args, **kwargs):
        eps = 1e-8

        r_hat = coords / torch.linalg.norm(coords, dim=-1, keepdim=True).clamp_min(eps)
        r_hat = r_hat.detach()

        b_cross_r = torch.cross(b, r_hat, dim=-1)

        b2 = torch.sum(b**2, dim=-1).detach().clamp_min(eps)

        radial_loss = torch.sum(b_cross_r**2, dim=-1) / b2

        return radial_loss


class PotentialLoss(BaseLoss):

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
        potential_loss = j.pow(2).sum(-1)

        return potential_loss


class EnergyGradientLoss(BaseLoss):

    def forward(self, b, jac_matrix, coords, *args, **kwargs):
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

        # dE/dr = dE/dx * dx/dr + dE/dy * dy/dr + dE/dz * dz/dr
        # theta is colatitude: x = r sin(t) cos(p), y = r sin(t) sin(p), z = r cos(t).
        coords_spherical = cartesian_to_spherical(coords, f=torch)
        t = coords_spherical[:, 1]
        p = coords_spherical[:, 2]
        dE_dr = (torch.sin(t) * torch.cos(p)) * dE_dx + \
                (torch.sin(t) * torch.sin(p)) * dE_dy + \
                torch.cos(t) * dE_dz

        energy_gradient_regularization = torch.relu(dE_dr)

        return energy_gradient_regularization


class EnergyLoss(BaseLoss):

    def __init__(self, base_radius, Mm_per_ds, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds

    def forward(self, b, coords, *args, **kwargs):
        energy_loss = torch.norm(b, dim=-1).pow(2)

        radius_weight = coords.pow(2).sum(-1).pow(0.5)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0)

        energy_regularization = energy_loss * radius_weight ** 2

        return energy_regularization


class NaNLoss(BaseLoss):

    def forward(self, boundary_b, b_true):
        if torch.isnan(b_true).sum() == 0:
            min_energy_NaNs_regularization = torch.zeros((1,), device=b_true.device)
        else:
            min_energy_NaNs_regularization = boundary_b[torch.isnan(b_true)].pow(2).sum() / torch.isnan(b_true).sum()
        return min_energy_NaNs_regularization


class LosTrvBoundaryLoss(BaseLoss):

    def forward(self, b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        b_pred = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        b_pred = img_to_los_trv_azi(b_pred, f=torch)

        b_los_trv_pred = b_pred[..., :2]
        b_los_trv_true = b_true[..., :2]

        # compute diff
        b_diff = b_los_trv_pred - b_los_trv_true
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        return b_diff


class AziBoundaryLoss(BaseLoss):

    def __init__(self, disambiguate=True, **kwargs):
        super().__init__(**kwargs)
        self.disambiguate = disambiguate

    def forward(self, b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        b_pred = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        b_pred = img_to_los_trv_azi(b_pred, f=torch)

        b_azi_true = b_true[..., 2] % torch.pi
        b_azi_pred = b_pred[..., 2] % torch.pi
        b_diff = (b_azi_pred - b_azi_true).pow(2) * b_true[..., 1]  # weight by transverse field
        b_diff = b_diff

        return b_diff


class LosTrvAziBoundaryLoss(BaseLoss):

    def __init__(self, ambiguous=False, los_weight=0.7, **kwargs):
        super().__init__(**kwargs)
        self.ambiguous = ambiguous
        assert los_weight >= 0 and los_weight <= 1, 'los_weight must be between 0 and 1'
        self.los_weight = los_weight

    def forward(self, b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        bxyz_pred = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        bxyz_true = los_trv_azi_to_img(b_true, f=torch)

        if self.ambiguous:
            # LOS component
            loss_bz = (bxyz_pred[..., 2] - bxyz_true[..., 2]).pow(2)

            # transverse components
            bxy_true = bxyz_true[..., :2]
            bxy_pred = bxyz_pred[..., :2]
            dot = (bxy_pred * bxy_true).sum(-1)
            loss_bxy = bxy_true.pow(2).sum(-1) + bxy_pred.pow(2).sum(-1) - 2 * torch.abs(dot)
        else:
            loss_bz = (bxyz_pred[..., 2] - bxyz_true[..., 2]).pow(2)
            loss_bxy = (bxyz_pred[..., :2] - bxyz_true[..., :2]).pow(2).sum(-1)

        loss = (loss_bz * self.los_weight + loss_bxy * (1 - self.los_weight)) * 2

        return loss


class LosBoundaryLoss(BaseLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        bxyz_pred = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b

        b_los_pred = bxyz_pred[..., 2]
        b_los_true = b_true[..., 0]

        # compute diff
        b_diff = (b_los_pred - b_los_true).pow(2)
        return b_diff


class BoundaryLoss(BaseLoss):

    def __init__(self, weights=None, **kwargs):
        super().__init__(**kwargs)
        weights = weights if weights is not None else [1.0, 1.0, 1.0]
        weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer('weights_buffer', weights)

    def forward(self, b, b_true, transform=None, b_err=None, weights=None, *args, **kwargs):
        # apply transforms
        transformed_b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        # compute diff
        b_err = b_err if b_err is not None else torch.zeros_like(b_true)
        b_diff = torch.clip(torch.abs(transformed_b - b_true) - b_err, 0)
        b_diff = torch.einsum('...i,i->...', b_diff.pow(2), self.weights_buffer)
        return b_diff


class WeightedHeightLoss(BaseLoss):

    def forward(self, coords, original_coords, height_range, *args, **kwargs):
        height_diff = torch.abs(coords[:, 2] - original_coords[:, 2])
        normalization = (height_range[:, 1] - height_range[:, 0]) + 1e-6
        height_regularization = torch.true_divide(height_diff, normalization)

        return height_regularization

class HeightLoss(BaseLoss):

    def forward(self, coords, original_coords, *args, **kwargs):
        height_diff = (coords[:, 2] - original_coords[:, 2]).pow(2)
        height_diff = height_diff / (original_coords[:, 2].pow(2) + 1e-7)

        return height_diff


class MinHeightLoss(BaseLoss):

    def forward(self, coords, *args, **kwargs):
        min_height_regularization = coords[:, 2].pow(2)
        return min_height_regularization

# mapping
loss_module_mapping = {'boundary': BoundaryLoss, 'boundary_los_trv': LosTrvBoundaryLoss,
                       'boundary_azi': AziBoundaryLoss,
                       'boundary_los_trv_azi': LosTrvAziBoundaryLoss, 'boundary_los': LosBoundaryLoss,
                       'divergence': DivergenceLoss, 'force_free': ForceFreeLoss, 'potential': PotentialLoss,
                       'weighted_height': WeightedHeightLoss, 'height': HeightLoss,
                       'NaNs': NaNLoss, 'radial': RadialLoss,
                       'min_height': MinHeightLoss, 'energy_gradient': EnergyGradientLoss, 'energy': EnergyLoss,
                       'sigma_j': SigmaJLoss
                       }
