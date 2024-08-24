import os

import numpy as np
import torch
from astropy.io import fits
from matplotlib import pyplot as plt
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from nf2.train.model import VectorPotentialModel, calculate_current


def load_cube(save_path, device=None, z=None, strides=1, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    z = z if z is not None else cube_shape[2]
    coords = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :z:strides], -1)
    return load_coords(model, state['spatial_norm'],
                       state['b_norm'], coords, device, **kwargs)


def load_height_surface(save_path, device=None, strides=1, batch_size=1000, progress=False, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = state['height_mapping_model']
    height_mapping = state['height_mapping']
    assert model is not None, 'Requires height mapping model!'
    model = nn.DataParallel(model)
    cube_shape = state['cube_shape']
    coords = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :len(height_mapping['z']):], -1)
    ranges = np.zeros((*coords.shape[:-1], 2))
    for i, (z, z_min, z_max) in enumerate(zip(height_mapping['z'], height_mapping['z_min'], height_mapping['z_max'])):
        coords[:, :, i, 2] = z
        ranges[:, :, i, 0] = z_min
        ranges[:, :, i, 1] = z_max

    # normalize and to tensor
    coords = torch.tensor(coords / state['spatial_norm'], dtype=torch.float32)
    coords_shape = coords.shape
    coords = coords.view((-1, 3))

    ranges = torch.tensor(ranges / state['spatial_norm'], dtype=torch.float32)
    ranges = ranges.view((-1, 2))

    slices = []
    it = range(int(np.ceil(coords.shape[0] / batch_size)))
    it = tqdm(it) if progress else it
    for k in it:
        coord = coords[k * batch_size: (k + 1) * batch_size]
        coord = coord.to(device)
        coord.requires_grad = True
        #
        r = ranges[k * batch_size: (k + 1) * batch_size]
        r = r.to(device)
        slices += [model(coord, r).detach().cpu()]

    slices = torch.cat(slices) * state['spatial_norm']
    slices = slices.view(*coords_shape).numpy()
    return slices


def load_height_cube(save_path, *args, device=None, strides=1, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    cube_shape = state['cube_shape']

    slice = load_height_surface(save_path, *args, device=device, strides=strides, **kwargs)

    cube = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :cube_shape[2]:strides], -1)[..., 2]
    contour_cube = np.zeros_like(cube)
    for i in range(slice.shape[2]):
        contour_cube[cube > slice[:, :, i:i + 1, 2]] = i + 1
    return contour_cube


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
    return load_coords(model, state['spatial_norm'],
                       state['b_norm'], coords, device, **kwargs)


def load_coords_from_state(save_path, coords, device=None, **kwargs):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    assert np.all(coords[..., 0] < cube_shape[0]), 'Invalid x coordinate, maximum is %d' % cube_shape[0]
    assert np.all(coords[..., 1] < cube_shape[1]), 'Invalid x coordinate, maximum is %d' % cube_shape[1]
    assert np.all(coords[..., 2] < cube_shape[2]), 'Invalid x coordinate, maximum is %d' % cube_shape[2]
    return load_coords(model, state['spatial_norm'], state['b_norm'], coords, device,
                       **kwargs)


def load_coords(model, spatial_norm, b_norm, coords, device, batch_size=1000, progress=False, compute_currents=False):
    def _load(coords):
        # normalize and to tensor
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords_shape = coords.shape
        coords = coords.reshape((-1, 3))

        cube = []
        j_cube = []
        it = range(int(np.ceil(coords.shape[0] / batch_size)))
        it = tqdm(it) if progress else it
        for k in it:
            model.zero_grad()
            coord = coords[k * batch_size: (k + 1) * batch_size]
            coord = coord.to(device)
            coord.requires_grad = True
            result = model(coord)
            b_batch = result['b']
            if compute_currents:
                j_batch = calculate_current(b_batch, coord)
                j_cube += [j_batch.detach().cpu()]
            cube += [b_batch.detach().cpu()]

        cube = torch.cat(cube)
        cube = cube.reshape(*coords_shape).numpy()
        b = cube * b_norm
        if compute_currents:
            j_cube = torch.cat(j_cube)
            j_cube = j_cube.reshape(*coords_shape).numpy()
            j = j_cube * b_norm / spatial_norm
            return b, j
        return b

    if (compute_currents or
            isinstance(model, VectorPotentialModel) or (
                    isinstance(model, nn.DataParallel) and isinstance(model.module, VectorPotentialModel))):
        return _load(coords)
    else:
        with torch.no_grad():
            return _load(coords)


def load_B_map(nf2_file, component=2):
    state = torch.load(nf2_file)
    meta = state['meta_data']
    mag_data = load_slice(nf2_file, z=0)
    mag_map = Map(mag_data[:, :, 0, component].T, meta)
    return mag_map


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
