import numpy as np
import torch
from astropy.io import fits
from astropy.nddata import block_reduce
from torch import nn

from nf2.loader.muram import read_muram_slice
from nf2.potential.potential_field import get_fft_potential_field
from astropy import units as u

class BaseScalingModule(nn.Module):
    """
    Base class for loss scaling modules.
    This class should be inherited by specific scaling modules.
    """

    def __init__(self, loss_ids, name=None):
        super().__init__()
        self.loss_ids = loss_ids
        self.name = name if name is not None else self.__class__.__name__

    def update(self, *args, **kwargs):
        return None


class ExponentialLossScalingModule(BaseScalingModule):
    """
    Loss scaling based on magnetic field B.
    """

    # Fit: B_norm(z) = 0.8920 * z^4 + -5.1084 * z^3 + 11.1580 * z^2 + -14.5714 * z + -4.7806
    def __init__(self, coeffs=[0.8920, 5.1084, 11.1580, 14.5714, 4.7806], *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(coeffs)
        self.a = coeffs[0]
        self.b = coeffs[1]
        self.c = coeffs[2]
        self.d = coeffs[3]
        self.e = coeffs[4]

    def forward(self, loss, state):
        """
        Forward pass for the B loss scaling module.
        This method scales the loss based on the estimated magnetic field B.
        :param loss: The loss tensor to be scaled.
        :param state: A dictionary containing the state information, including 'coords' and 'b
        :return:
        """
        coords = state['coords']
        z = coords[..., 2]  # alternative detach() here

        scaling = torch.exp(
            self.a * z ** 4 +
            self.b * z ** 3 +
            self.c * z ** 2 +
            self.d * z +
            self.e
        )
        scaling = scaling.detach()  # No gradient through scaling --> relevant for coordinate transforms
        scaled_loss = loss / (scaling + 1e-6)  # Avoid division by zero

        return scaled_loss


class PotentialFitLossScalingModule(BaseScalingModule):
    """
    Loss scaling based on magnetic field B.
    """

    def __init__(self, ref_file, Mm_per_pixel, Mm_per_ds, Gauss_per_dB, binning=4, power=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        coeffs = self._fit_coeffs(ref_file, binning, Mm_per_pixel, Mm_per_ds, Gauss_per_dB)
        print(f'Fitted coefficients from {ref_file}:', coeffs)
        self.a = coeffs[0]
        self.b = coeffs[1]
        self.c = coeffs[2]
        self.power = power

    def _fit_coeffs(self, ref_file, binning, Mm_per_pixel, Mm_per_ds, Gauss_per_dB):
        bz = self._load_data(ref_file)

        bz = block_reduce(bz, block_size=(binning, binning), func=np.mean)
        Mm_per_pixel = Mm_per_pixel * binning

        print('Loading reference potential field for loss scaling fit...')
        potential_field = get_fft_potential_field(bz, max(bz.shape))
        potential_field = potential_field / Gauss_per_dB  # normalize to model units

        # Convert to Mm
        z = np.linspace(0, potential_field.shape[2] - 1, potential_field.shape[2]) * Mm_per_pixel
        z = z / Mm_per_ds  # normalize to model units
        b_norm = np.linalg.norm(potential_field, axis=-1).mean((0, 1))

        # use log units
        log_b_norm = np.log(b_norm + 1e-8)
        t = np.log(1 + z)

        #########################################################
        # Fit polynomial
        coeffs = np.polyfit(t, log_b_norm, deg=2)
        return coeffs

    def _load_data(self, ref_file):
        if ref_file.endswith('.fits'):
            bz = fits.getdata(ref_file).T  # (x, y)
        else:
            sl, Nvar, shape, time = read_muram_slice(ref_file)
            bz = sl[5, :, :] * np.sqrt(4 * np.pi)
        return bz

    def forward(self, loss, state):
        """
        Forward pass for the B loss scaling module.
        This method scales the loss based on the estimated magnetic field B.
        :param loss: The loss tensor to be scaled.
        :param state: A dictionary containing the state information, including 'coords' and 'b
        :return:
        """
        coords = state['coords']
        z = coords[..., 2]  # alternative detach() here
        t = torch.log(1 + z)

        scaling = torch.exp(self.a * t ** 2 + self.b * t + self.c)
        scaling = scaling.pow(self.power).detach()  # No gradient through scaling --> relevant for coordinate transforms
        scaled_loss = loss / (scaling + 1e-6)  # Avoid division by zero

        return scaled_loss


class BHeightLossScalingModule(BaseScalingModule):
    """
    Loss scaling based on magnetic field B.
    """

    def __init__(self, power=2, detach=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
        self.detach = detach

    def forward(self, loss, state):
        """
        Forward pass for the B loss scaling module.
        This method scales the loss based on the estimated magnetic field B.
        :param loss: The loss tensor to be scaled.
        :param state: A dictionary containing the state information, including 'coords' and 'b
        :return:
        """
        grouped_coords = state['grouped_coords']  # (angular samples, radial samples, 3)
        b = state['b']  # (batch_size, 3)

        b = b.reshape(grouped_coords.shape[0], grouped_coords.shape[1], b.shape[-1])
        loss = loss.reshape(grouped_coords.shape[0], grouped_coords.shape[1])

        b_norm = (b.norm(dim=-1) + 1e-6).pow(self.power)  # (angular samples, radial samples)
        scaling = b_norm.mean(0, keepdim=True)  # mean across angular samples
        if self.detach:
            scaling = scaling.detach()  # No gradient through scaling
        scaled_loss = loss / scaling

        scaled_loss = scaled_loss.reshape(-1)

        return scaled_loss

class RadialLossScalingModule(BaseScalingModule):
    """
    Loss scaling based on radial distance r.
    """

    def __init__(self, Mm_per_ds, base_radius=1.1, max_radius=None, **kwargs):
        super().__init__(**kwargs)
        self.base_radius = (base_radius * u.solRad).to_value(u.Mm) / Mm_per_ds
        self.max_radius = None if max_radius is None else (max_radius * u.solRad).to_value(u.Mm) / Mm_per_ds

        if self.max_radius is not None and self.max_radius <= self.base_radius:
            raise ValueError('max_radius must be greater than base_radius for radial loss scaling.')

    def forward(self, loss, state):
        """
        Forward pass for the radial loss scaling module.
        This method scales the loss based on the radial distance r.
        :param loss: The loss tensor to be scaled.
        :param state: A dictionary containing the state information, including 'coords'.
        :return:
        """
        coords = state['coords']

        radius_weight = coords.pow(2).sum(-1).pow(0.5)
        radius_weight = radius_weight - self.base_radius

        if self.max_radius is not None:
            radius_weight = radius_weight / (self.max_radius - self.base_radius)
            radius_weight = torch.clip(radius_weight, min=0, max=1)
        else:
            radius_weight = torch.clip(radius_weight, min=0)

        scaled_loss = loss * radius_weight

        return scaled_loss
