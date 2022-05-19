import numpy as np

from nf2.potential.potential_field import get_potential
from nf2.train.metric import energy


def get_free_mag_energy(b, **kwargs):
    potential = get_potential(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3), **kwargs)
    b_potential = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)
    #
    free_energy = energy(b) - energy(b_potential)
    return free_energy
