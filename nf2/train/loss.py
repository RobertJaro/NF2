import torch
from torch import nn

from nf2.data.util import cartesian_to_spherical, img_to_los_trv_azi, los_trv_azi_to_img
from astropy import units as u

class BaseLoss(nn.Module):

    def __init__(self, name, ds_id, **kwargs):
        super().__init__()
        self.name = name
        self.ds_id = ds_id


class ForceFreeLoss(BaseLoss):

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
        # assure that the normalization does not influence the loss
        normalization = (torch.sum(b ** 2, dim=-1) + 1e-7).detach()
        force_free_loss = torch.sum(jxb ** 2, dim=-1) / normalization
        #
        return force_free_loss.mean()


class MagnetoStaticLoss(BaseLoss):

    def forward(self, b, jac_matrix, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dx = jac_matrix[:, 1, 0]
        dBz_dx = jac_matrix[:, 2, 0]
        dP_dx = jac_matrix[:, 3, 0]
        dBx_dy = jac_matrix[:, 0, 1]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dy = jac_matrix[:, 2, 1]
        dP_dy = jac_matrix[:, 3, 1]
        dBx_dz = jac_matrix[:, 0, 2]
        dBy_dz = jac_matrix[:, 1, 2]
        dBz_dz = jac_matrix[:, 2, 2]
        dP_dz = jac_matrix[:, 3, 2]
        #
        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        #
        j = torch.stack([rot_x, rot_y, rot_z], -1)
        jxb = torch.cross(j, b, -1)
        grad_P = torch.stack([dP_dx, dP_dy, dP_dz], -1)
        #
        equation = jxb - grad_P
        # assure that the normalization does not influence the loss
        normalization = (torch.sum(b ** 2, dim=-1) + 1e-7).detach()
        loss = torch.sum(equation ** 2, dim=-1) / normalization
        #
        return loss.mean()


class DivergenceLoss(BaseLoss):

    def forward(self, jac_matrix, *args, **kwargs):
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dz = jac_matrix[:, 2, 2]

        divergence_loss = (dBx_dx + dBy_dy + dBz_dz) ** 2

        return divergence_loss.mean()


class RadialLoss(BaseLoss):

    def __init__(self, base_radius, Mm_per_ds, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds

    def forward(self, b, coords, *args, **kwargs):
        # radial regularization --> vanishing phi and theta components
        radial_regularization = torch.norm(torch.cross(b, coords, dim=-1), dim=-1)

        radius_weight = torch.sqrt(torch.sum(coords ** 2, dim=-1) + 1e-7)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0)

        normalization = torch.norm(b, dim=-1) * torch.norm(coords, dim=-1) + 1e-7
        radial_regularization = radial_regularization / normalization

        radial_regularization = (radial_regularization * radius_weight).mean()

        return radial_regularization

class PotentialLoss(BaseLoss):

    def __init__(self, base_radius, Mm_per_ds, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds
        self.solar_radius = (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

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

        if self.base_radius is not None:
            radius = coords.pow(2).sum(-1).pow(0.5) + 1e-7
            radius_weight = torch.clip(radius - self.base_radius, min=0) / self.solar_radius # normalize to solar radius
            potential_loss *= radius_weight ** 2

        return potential_loss.mean()


class EnergyGradientLoss(BaseLoss):

    def __init__(self, base_radius, Mm_per_ds, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds
        self.solar_radius = (1 * u.solRad).to_value(u.Mm) / Mm_per_ds
        # self.asinh_stretch = nn.Parameter(torch.tensor(np.arcsinh(1e3), dtype=torch.float32), requires_grad=False)

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

        coords_spherical = cartesian_to_spherical(coords, f=torch)
        t = coords_spherical[:, 1]
        p = coords_spherical[:, 2]
        dE_dr = (torch.sin(t) * torch.cos(p)) * dE_dx + \
                (torch.sin(t) * torch.sin(p)) * dE_dy + \
                torch.cos(p) * dE_dz

        radius_weight = coords.pow(2).sum(-1).pow(0.5)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0) / self.solar_radius # normalize to solar radius

        sampled_dE_dr = dE_dr
        energy_gradient_regularization = torch.relu(sampled_dE_dr) * radius_weight ** 2
        energy_gradient_regularization = energy_gradient_regularization.mean()

        return energy_gradient_regularization

class EnergyLoss(BaseLoss):

    def __init__(self, base_radius, Mm_per_ds, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds

    def forward(self, b, coords, *args, **kwargs):
        energy_loss = torch.norm(b, dim=-1).pow(2).mean()

        radius_weight = coords.pow(2).sum(-1).pow(0.5)
        radius_weight = torch.clip(radius_weight - self.base_radius, min=0)

        energy_regularization = energy_loss * radius_weight ** 2
        energy_regularization = energy_regularization.mean()

        return energy_regularization

class NaNLoss(BaseLoss):

    def forward(self, boundary_b, b_true):
        if torch.isnan(b_true).sum() == 0:
            min_energy_NaNs_regularization = torch.zeros((1,), device=b_true.device)
        else:
            min_energy_NaNs_regularization = boundary_b[torch.isnan(b_true)].pow(2).sum() / torch.isnan(b_true).sum()
        return min_energy_NaNs_regularization


class LosTrvAziBoundaryLoss(BaseLoss):

    def __init__(self, disambiguate=True, **kwargs):
        super().__init__(**kwargs)
        self.disambiguate = disambiguate

    def forward(self, b, b_true, transform=None, *args, **kwargs):
        # apply transforms
        b_pred = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        b_pred = img_to_los_trv_azi(b_pred, f=torch)

        bxyz_true = los_trv_azi_to_img(b_true, ambiguous=True, f=torch)
        bxyz_pred = los_trv_azi_to_img(b_pred, ambiguous=True, f=torch)

        # compute diff
        b_diff = bxyz_pred - bxyz_true
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        return b_diff


class BoundaryLoss(BaseLoss):

    def forward(self, b, b_true, transform=None, b_err=None, *args, **kwargs):
        # apply transforms
        transformed_b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        # compute diff
        b_err = b_err if b_err is not None else torch.zeros_like(b_true)
        b_diff = torch.clip(torch.abs(transformed_b - b_true) - b_err, 0)
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        assert not torch.isnan(b_diff), 'b_diff is nan'
        return b_diff


class HeightLoss(BaseLoss):

    def forward(self, coords, original_coords, height_range, *args, **kwargs):
        height_diff = torch.abs(coords[:, 2] - original_coords[:, 2])
        normalization = (height_range[:, 1] - height_range[:, 0]) + 1e-6
        height_regularization = torch.true_divide(height_diff, normalization).mean()

        return height_regularization


class MinHeightLoss(BaseLoss):

    def forward(self, b_true, coords, *args, **kwargs):
        min_height_regularization = torch.abs(coords[:, 2])
        # min_height_regularization = min_height_regularization / (torch.norm(b_true, dim=-1) + 1e-7)
        min_height_regularization = min_height_regularization.mean()
        return min_height_regularization


class FluxPreservationLoss(BaseLoss):

    def forward(self, dflux_dr, *args, **kwargs):
        flux_preservation_loss = dflux_dr.pow(2).mean()
        return flux_preservation_loss


class SphericalTransform(nn.Module):

    def forward(self, b, b_true, transform=None):
        b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b
        return b, b_true
