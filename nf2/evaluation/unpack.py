import os

import numpy as np
import torch
from astropy.io import fits
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def load_cube(save_path, device=None, z=None, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    z = z if z is not None else cube_shape[2]
    coords = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], :z], -1)
    return load_coords(model, cube_shape, state['spatial_normalization'],
                       state['normalization'], coords, device, **kwargs)

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
    return load_coords(model, cube_shape, state['spatial_normalization'],
                       state['normalization'], coords, device, **kwargs)


def load_coords_from_state(save_path, coords, device=None, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    return load_coords(model, cube_shape, state['spatial_normalization'], state['normalization'], coords, device,
                       **kwargs)


def load_coords(model, cube_shape, spatial_norm, b_norm, coords, batch_size=1000, progress=False):
    assert np.all(coords[..., 0] < cube_shape[0]), 'Invalid x coordinate, maximum is %d' % cube_shape[0]
    assert np.all(coords[..., 1] < cube_shape[1]), 'Invalid x coordinate, maximum is %d' % cube_shape[1]
    assert np.all(coords[..., 2] < cube_shape[2]), 'Invalid x coordinate, maximum is %d' % cube_shape[2]

    with torch.no_grad():
        # normalize and to tensor
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords_shape = coords.shape
        coords = coords.view((-1, 3))

        cube = []
        it = range(int(np.ceil(coords.shape[0] / batch_size)))
        it = tqdm(it) if progress else it
        for k in it:
            coord = coords[k * batch_size: (k + 1) * batch_size]
            coord = coord.to(model.device)
            cube += [model(coord).detach().cpu()]

        cube = torch.cat(cube)
        cube = cube.view(*coords_shape).numpy()
    b = cube * b_norm
    return b


def save_fits(vec, path, prefix):
    hdu = fits.PrimaryHDU(vec[..., 0])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_Bx.fits' % prefix))
    hdu = fits.PrimaryHDU(vec[..., 1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_By.fits' % prefix))
    hdu = fits.PrimaryHDU(vec[..., 2])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_Bz.fits' % prefix))


def save_slice(b, file_path, v_min_max=None):
    v_min_max = np.abs(b).max() if v_min_max is None else v_min_max
    plt.imsave(file_path, b.transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max, origin='lower')
