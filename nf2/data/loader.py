import glob
import os

import numpy as np
import torch
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from sunpy.map import Map, all_coordinates_from_map

from nf2.potential.potential_field import get_potential_boundary, get_potential_top


def prep_b_data(b_cube, error_cube, height,
                potential_boundary=True, potential_strides=4):
    # load coordinates
    mf_coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :1], -1)
    # flatten data
    mf_coords = mf_coords.reshape((-1, 3))
    mf_values = b_cube.reshape((-1, 3))
    mf_err = error_cube.reshape((-1, 3))
    # load potential field
    if potential_boundary:
        pf_coords, pf_err, pf_values = load_potential_field_data(b_cube, height, potential_strides)
        # concatenate pf data points
        coords = np.concatenate([pf_coords, mf_coords])
        values = np.concatenate([pf_values, mf_values])
        err = np.concatenate([pf_err, mf_err])
    else:
        coords = mf_coords
        values = mf_values
        err = mf_err

    coords = coords.astype(np.float32)
    values = values.astype(np.float32)
    err = err.astype(np.float32)

    return coords, values, err

def load_potential_field_data(hmi_cube, height, reduce, only_top=False, pf_error=0.0, **kwargs):
    if reduce > 1:
        hmi_cube = block_reduce(hmi_cube, (reduce, reduce, 1), func=np.mean)
        height = height // reduce
    pf_batch_size = int(1024 * 512 ** 2 / np.prod(hmi_cube.shape[:2]))  # adjust batch to AR size
    pf_coords, pf_values = get_potential_top(hmi_cube[:, :, 2], height, batch_size=pf_batch_size, **kwargs) \
        if only_top else get_potential_boundary(hmi_cube[:, :, 2], height, batch_size=pf_batch_size, **kwargs)
    pf_values = np.array(pf_values, dtype=np.float32)
    pf_coords = np.array(pf_coords, dtype=np.float32) * reduce  # expand to original coordinate spacing
    pf_err = np.ones_like(pf_values) * pf_error
    return pf_coords, pf_err, pf_values


def _plot_data(error_cube, n_hmi_cube, plot_path, b_norm):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(n_hmi_cube[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    axs[1].imshow(n_hmi_cube[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    axs[2].imshow(n_hmi_cube[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    plt.savefig(os.path.join(plot_path, 'b.jpg'))
    plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(error_cube[..., 0].transpose(), vmin=0, cmap='gray', origin='lower')
    axs[1].imshow(error_cube[..., 1].transpose(), vmin=0, cmap='gray', origin='lower')
    axs[2].imshow(error_cube[..., 2].transpose(), vmin=0, cmap='gray', origin='lower')
    plt.savefig(os.path.join(plot_path, 'b_err.jpg'))
    plt.close()


class RandomSphericalCoordinateSampler():

    def __init__(self, height, batch_size, cuda=True):
        self.height = height
        self.batch_size = batch_size
        self.float_tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def load_sample(self):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        random_coords[:, 0] = random_coords[:, 0] * 2 * np.pi  # phi [0, 2pi]
        random_coords[:, 1] = random_coords[:, 1] * np.pi  # theta [0, pi]
        random_coords[:, 2] = 1 + random_coords[:, 2] * (self.height - 1)  # r [1, height]
        random_coords = self.to_cartesian(random_coords)
        return random_coords

    def to_cartesian(self, v):
        sin = torch.sin
        cos = torch.cos
        p, t, r = v[..., 0], v[..., 1], v[..., 2]
        x = r * sin(t) * cos(p)
        y = r * sin(t) * sin(p)
        z = r * cos(t)
        return torch.stack([x, y, z], -1)
