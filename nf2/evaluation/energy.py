import numpy as np

from nf2.potential.potential_field import get_potential_field, get_fft_potential_field
from nf2.evaluation.metric import energy


def get_free_mag_energy_direct(b, **kwargs):
    b_potential = get_potential_field(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3), **kwargs)
    return energy(b) - energy(b_potential)


def get_free_mag_energy_fft(b, scale=1, **kwargs):
    b_potential = get_fft_potential_field(b[:, :, 0, 2], b.shape[2], scale=scale, **kwargs)
    return energy(b) - energy(b_potential)


def get_free_mag_energy(b, method='fft', **kwargs):
    method = method.lower()
    if method == 'fft':
        return get_free_mag_energy_fft(b, **kwargs)
    if method in {'direct', 'green', 'greens'}:
        return get_free_mag_energy_direct(b, **kwargs)
    raise ValueError("Free-energy potential method must be 'fft' or 'direct'.")
