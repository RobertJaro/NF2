import numpy as np

from nf2.potential.potential_field import get_potential_field, get_fft_potential_field
from nf2.evaluation.metric import energy


def get_free_mag_energy(b, **kwargs):
    b_potential = get_potential_field(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3), **kwargs)
    #
    free_energy = energy(b) - energy(b_potential)
    return free_energy

def get_free_mag_energy_fft(b, **kwargs):
    b_potential = get_fft_potential_field(b[:, :, 0, 2], b.shape[2], scale=1)
    #
    free_energy = energy(b) - energy(b_potential)
    return free_energy