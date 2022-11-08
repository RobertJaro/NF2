import os

import numpy as np
import torch
from astropy.io import fits
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from nf2.train.model import VectorPotentialModel


def load_cube(save_path, device=None, z=None, strides=1, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    z = z if z is not None else cube_shape[2]
    coords = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :z:strides], -1)
    return load_coords(model, cube_shape, state['spatial_norm'],
                       state['b_norm'], coords, device, **kwargs)

def load_shape(save_path, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    return state['cube_shape']


def load_slice(save_path, z=0, device=None, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    z = z if z is not None else cube_shape[2]
    coords = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], z:z + 1], -1)
    return load_coords(model, cube_shape, state['spatial_norm'],
                       state['b_norm'], coords, device, **kwargs)


def load_coords_from_state(save_path, coords, device=None, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    return load_coords(model, cube_shape, state['spatial_norm'], state['b_norm'], coords, device,
                       **kwargs)


def load_coords(model, cube_shape, spatial_norm, b_norm, coords, device, batch_size=1000, progress=False):
    assert np.all(coords[..., 0] < cube_shape[0]), 'Invalid x coordinate, maximum is %d' % cube_shape[0]
    assert np.all(coords[..., 1] < cube_shape[1]), 'Invalid x coordinate, maximum is %d' % cube_shape[1]
    assert np.all(coords[..., 2] < cube_shape[2]), 'Invalid x coordinate, maximum is %d' % cube_shape[2]

    def _load(coords):
        # normalize and to tensor
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords_shape = coords.shape
        coords = coords.view((-1, 3))

        cube = []
        it = range(int(np.ceil(coords.shape[0] / batch_size)))
        it = tqdm(it) if progress else it
        for k in it:
            coord = coords[k * batch_size: (k + 1) * batch_size]
            coord = coord.to(device)
            coord.requires_grad = True
            cube += [model(coord).detach().cpu()]

        cube = torch.cat(cube)
        cube = cube.view(*coords_shape).numpy()
        b = cube * b_norm
        return b
    if isinstance(model, VectorPotentialModel) or \
            (isinstance(model, nn.DataParallel) and isinstance(model.module, VectorPotentialModel)):
        return _load(coords)
    else:
        with torch.no_grad():
            return _load(coords)



def save_fits(vec, path, prefix, meta_info={}):
    hdu = fits.PrimaryHDU(vec[..., 0].T)
    for i, v in meta_info.items():
        hdu.header[i] = v
    hdul = fits.HDUList([hdu])
    x_path = os.path.join(path, '%s_Bx.fits' % prefix)
    hdul.writeto(x_path)

    hdu = fits.PrimaryHDU(vec[..., 1].T)
    for i, v in meta_info.items():
        hdu.header[i] = v
    hdul = fits.HDUList([hdu])
    y_path = os.path.join(path, '%s_By.fits' % prefix)
    hdul.writeto(y_path)

    hdu = fits.PrimaryHDU(vec[..., 2].T)
    for i, v in meta_info.items():
        hdu.header[i] = v
    hdul = fits.HDUList([hdu])
    z_path = os.path.join(path, '%s_Bz.fits' % prefix)
    hdul.writeto(z_path)
    return x_path, y_path, z_path


def save_slice(b, file_path, v_min_max=None):
    v_min_max = np.abs(b).max() if v_min_max is None else v_min_max
    plt.imsave(file_path, b.transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max, origin='lower')
