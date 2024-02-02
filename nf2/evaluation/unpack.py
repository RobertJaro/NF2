import os
from multiprocessing import Pool

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from sunpy.coordinates import frames
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from nf2.data.util import spherical_to_cartesian, cartesian_to_spherical, vector_cartesian_to_spherical
from nf2.train.model import VectorPotentialModel, calculate_current, FluxModel


class BaseOutput():

    def __init__(self, checkpoint, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = torch.load(checkpoint, map_location=device)
        model = self.state['model']
        self._requires_grad = isinstance(model, VectorPotentialModel) or isinstance(model, FluxModel)
        self.model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        self.spatial_norm = 1
        self.device = device

    @property
    def G_per_dB(self):
        return self.state['data']['G_per_dB']

    @property
    def m_per_ds(self):
        return self.state['data']['Mm_per_ds']

    def load_coords(self, coords, batch_size=int(2 ** 12), progress=False, compute_currents=False):
        def _load(coords):
            # normalize and to tensor
            coords = torch.tensor(coords / self.spatial_norm, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            cube = []
            j_cube = []
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it) if progress else it
            for k in it:
                self.model.zero_grad()
                coord = coords[k * batch_size: (k + 1) * batch_size]
                coord = coord.to(self.device)
                coord.requires_grad = True
                result = self.model(coord)
                b_batch = result['b']
                if compute_currents:
                    j_batch = calculate_current(b_batch, coord)
                    j_cube += [j_batch.detach().cpu()]
                cube += [b_batch.detach().cpu()]

            cube = torch.cat(cube)
            cube = cube.reshape(*coords_shape).numpy()
            b = cube * self.G_per_dB

            model_out = {'B': b}
            if compute_currents:
                j_cube = torch.cat(j_cube)
                j_cube = j_cube.reshape(*coords_shape).numpy()
                j = j_cube * self.G_per_dB / self.m_per_ds
                model_out['J'] = j
            return model_out

        if compute_currents or self._requires_grad:
            return _load(coords)
        else:
            with torch.no_grad():
                return _load(coords)


class FieldLine:

    def __init__(self, coords):
        self.coords = coords



class SphericalOutput(BaseOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state['data']['type'] == 'spherical', 'Requires spherical NF2 data!'

        self.radius_range = self.state['data']['radius_range']

    def trace(self, seeds):
        seeds = self._skycoords_to_cartesian(seeds)
        field_lines = [self._get_field_line(seed) for seed in tqdm(seeds)]
        return field_lines

    def _get_field_line(self, seed, atol = 1e-4,  rtol = 1e-4):
            forward = solve_ivp(self._get_direction, (0, 1e1), seed, method='RK23', rtol=rtol, atol=atol, args=(1,))
            backward = solve_ivp(self._get_direction, (0, 1e1), seed, method='RK23', rtol=rtol, atol=atol, args=(-1,))
            field_line = np.concatenate([forward.y, backward.y[::-1]], axis=-1)
            field_line = field_line.T
            return field_line

    def _get_direction(self, t, coord, direction):
        radius = np.linalg.norm(coord, axis=-1)
        if radius < (self.radius_range[0].to_value(u.solRad) - 1e-2) or radius > self.radius_range[1].to_value(u.solRad):
            return np.array([0, 0, 0])
        coord = np.reshape(coord, (1, 3))
        b = self.load_coords(coord)['B']
        b = b[0] # remove batch dimension
        return np.sign(direction) * b / np.linalg.norm(b, axis=-1, keepdims=True)

    def load(self,
             radius_range: u.Quantity = None,
             latitude_range: u.Quantity = (-np.pi / 2, np.pi / 2) * u.rad,
             longitude_range: u.Quantity = (0, 2 * np.pi),
             resolution: u.Quantity = 64 * u.pix / u.solRad, **kwargs):
        radius_range = radius_range if radius_range is not None else self.radius_range
        latitude_range += np.pi / 2 * u.rad # transform coordinate frame
        spherical_bounds = np.stack(
            np.meshgrid(np.linspace(radius_range[0].to_value(u.solRad), radius_range[1].to_value(u.solRad), 50),
                        np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), 50),
                        np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), 50), indexing='ij'), -1)

        cartesian_bounds = spherical_to_cartesian(spherical_bounds)
        x_min, x_max = cartesian_bounds[..., 0].min(), cartesian_bounds[..., 0].max()
        y_min, y_max = cartesian_bounds[..., 1].min(), cartesian_bounds[..., 1].max()
        z_min, z_max = cartesian_bounds[..., 2].min(), cartesian_bounds[..., 2].max()

        res = resolution.to_value(u.pix / u.solRad)
        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * res)),
                        np.linspace(y_min, y_max, int((y_max - y_min) * res)),
                        np.linspace(z_max, z_min, int((z_max - z_min) * res)), indexing='ij'), -1)
        # flipped z axis
        spherical_coords = cartesian_to_spherical(coords)
        condition = (spherical_coords[..., 0] * u.solRad >= radius_range[0]) & (spherical_coords[..., 0] * u.solRad < radius_range[1]) \
                    & (spherical_coords[..., 1] * u.rad > latitude_range[0]) & (spherical_coords[..., 1] * u.rad < latitude_range[1]) \
            # & (spherical_coords[..., 2] > longitude_range[0]) & (spherical_coords[..., 2] < longitude_range[1])
        sub_coords = coords[condition]

        cube_shape = coords.shape[:-1]
        model_out = self.load_coords(sub_coords, compute_currents=True, **kwargs)

        # flip z axis
        for o in model_out.values():
            o[..., 2] *= -1

        sub_b = model_out['B']
        b = np.zeros(cube_shape + (3,))
        b[condition] = sub_b
        b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

        sub_j = model_out['J']
        j = np.zeros(cube_shape + (3,))
        j[condition] = sub_j

        return {'B': b, 'B_rtp': b_spherical, 'J': j, 'coords': coords, 'spherical_coords': spherical_coords}

    def load_spherical_coords(self, spherical_coords: SkyCoord):
        cartesian_coords = self._skycoords_to_cartesian(spherical_coords)

        return self.load_coords(cartesian_coords)

    def _skycoords_to_cartesian(self, spherical_coords):
        spherical_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
        r = spherical_coords.radius
        r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
        spherical_coords = np.stack([
            r.to(u.solRad).value,
            np.pi / 2 + spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        return cartesian_coords


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
                    isinstance(model, nn.DataParallel) and isinstance(model.module, VectorPotentialModel)) or \
            isinstance(model, FluxModel) or (
                    isinstance(model, nn.DataParallel) and isinstance(model.module, FluxModel))):
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
