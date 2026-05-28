import os
import glob
import logging
import re
import multiprocessing as mp
import warnings
from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import rank_zero_only
from sunpy.coordinates import frames
from sunpy import log as sunpy_log
from sunpy.map import Map, all_coordinates_from_map
from sunpy.util.exceptions import SunpyMetadataWarning

from nf2.data.dataset import NF2Dataset, RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset, \
    RandomRadialGroupedCoordinateDataset, TensorsDataset
from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian, cartesian_to_spherical_matrix
from nf2.loader.base import BaseDataModule, DEFAULT_NUM_WORKERS


SOLAR_RADIUS_Mm = (1 * u.solRad).to_value(u.Mm)


def spherical_coord_scale(Mm_per_ds):
    return SOLAR_RADIUS_Mm / Mm_per_ds


@contextmanager
def _suppress_sunpy_metadata_messages():
    logger = logging.getLogger('sunpy.map.mapbase')
    previous_level = logger.level
    previous_astropy_level = astropy_log.level
    previous_sunpy_level = sunpy_log.level
    logger.setLevel(logging.ERROR)
    astropy_log.setLevel('ERROR')
    sunpy_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SunpyMetadataWarning)
            yield
    finally:
        logger.setLevel(previous_level)
        astropy_log.setLevel(previous_astropy_level)
        sunpy_log.setLevel(previous_sunpy_level)


def _map(*args, **kwargs):
    with _suppress_sunpy_metadata_messages():
        return Map(*args, **kwargs)


def _all_coordinates_from_map(smap):
    with _suppress_sunpy_metadata_messages():
        return all_coordinates_from_map(smap)


def _carrington_coordinates_from_map(smap):
    with _suppress_sunpy_metadata_messages():
        return all_coordinates_from_map(smap).transform_to(frames.HeliographicCarrington)


def _mu_filter_mask(smap, mu_filter):
    if mu_filter is None:
        return None
    if isinstance(mu_filter, (int, float)):
        mu_filter = {'min': mu_filter}

    coords = _all_coordinates_from_map(smap)
    rho = np.hypot(coords.Tx.to_value(u.rad), coords.Ty.to_value(u.rad))
    with _suppress_sunpy_metadata_messages():
        rsun = smap.rsun_obs.to_value(u.rad)
    radial_distance = rho / rsun

    with np.errstate(invalid='ignore'):
        mu = np.sqrt(1 - radial_distance ** 2)

    mask = ~np.isfinite(mu) | (radial_distance > 1)
    if 'min' in mu_filter:
        mask |= mu < mu_filter['min']
    if 'max' in mu_filter:
        mask |= mu > mu_filter['max']
    return mask


def _reference_map_lon_lat(reference_map):
    carrington_coords = _carrington_coordinates_from_map(reference_map)
    with _suppress_sunpy_metadata_messages():
        center_lon = reference_map.center.transform_to(frames.HeliographicCarrington).lon
    longitude_deg = carrington_coords.lon.wrap_at(center_lon + 180 * u.deg).to_value(u.deg)
    latitude_deg = carrington_coords.lat.to_value(u.deg)
    return longitude_deg, latitude_deg


def _reference_map_coordinate_bounds(reference_map, mu_filter=None, name='reference map'):
    longitude_deg, latitude_deg = _reference_map_lon_lat(reference_map)
    valid = np.isfinite(reference_map.data) & np.isfinite(longitude_deg) & np.isfinite(latitude_deg)

    mu_mask = _mu_filter_mask(reference_map, mu_filter)
    if mu_mask is not None:
        valid &= ~mu_mask

    if not np.any(valid):
        raise ValueError(f'No valid coordinates found in {name}.')

    return {
        'latitude_range': [float(np.nanmin(latitude_deg[valid])),
                           float(np.nanmax(latitude_deg[valid]))],
        'longitude_range': [float(np.nanmin(longitude_deg[valid])),
                            float(np.nanmax(longitude_deg[valid]))],
    }


class SphericalSliceDataset(TensorsDataset):

    def __init__(self, b, coords, spherical_coords, Gauss_per_dB, Mm_per_ds,
                 b_err=None, transform=None,
                 plot_overview=True, strides=1, **kwargs):
        ds_name = kwargs.get('ds_name')
        if plot_overview:
            self._plot(b, coords, spherical_coords, transform, ds_name)
        if strides > 1:
            b = b[::strides, ::strides]
            coords = coords[::strides, ::strides]
            spherical_coords = spherical_coords[::strides, ::strides]
            b_err = b_err[::strides, ::strides] if b_err is not None else None
            transform = transform[::strides, ::strides] if transform is not None else None
        self.cube_shape = b.shape[:-1]
        # flatten data
        b = b.reshape((-1, 3))
        coords = coords.reshape((-1, 3))
        if b_err is not None:
            b_err = b_err.reshape((-1, 3))
        if transform is not None:
            transform = transform.reshape((-1, 3, 3))

        # normalize data
        b /= Gauss_per_dB
        b_err = b_err / Gauss_per_dB if b_err is not None else None
        self.Mm_per_ds = Mm_per_ds
        self.coord_scale = spherical_coord_scale(Mm_per_ds)
        self.radius_range = np.array([
            np.nanmin(spherical_coords[..., 0]),
            np.nanmax(spherical_coords[..., 0]),
        ], dtype=np.float32)
        self.cartesian_radius_range = np.array([
            np.nanmin(np.linalg.norm(coords, axis=-1)),
            np.nanmax(np.linalg.norm(coords, axis=-1)),
        ], dtype=np.float32)
        if not np.allclose(self.cartesian_radius_range, self.radius_range, rtol=1e-4, atol=1e-5, equal_nan=True):
            raise ValueError(
                'Spherical map coordinate scaling is inconsistent: Cartesian coordinate radius does not match '
                'the spherical-coordinate radius before model-unit normalization.'
            )
        coords = coords * self.coord_scale

        nan_mask = np.isnan(b).any(-1)
        coords[nan_mask] = np.nan
        b[nan_mask] = np.nan

        tensors = {'coords': coords,
                   'b_true': b}
        if transform is not None:
            transform[nan_mask] = np.nan
            tensors['transform'] = transform
        if b_err is not None:
            b_err[nan_mask] = np.nan
            tensors['b_err'] = b_err

        super().__init__(tensors, **kwargs)

    @rank_zero_only
    def _plot(self, b, coords, spherical_coords, transform, ds_name=None):
        log_prefix = f'{ds_name}/' if ds_name is not None else ''
        b_min_max = 200

        fig, axs = plt.subplots(3, 2, figsize=(8, 8))

        ax = axs[0, 0]
        im = ax.imshow(spherical_coords[..., 0].transpose(), origin='lower')
        ax.set_title('$r$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(spherical_coords[..., 1].transpose(), origin='lower')
        ax.set_title(r'$\theta$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[2, 0]
        im = ax.imshow(spherical_coords[..., 2].transpose(), origin='lower')
        ax.set_title('$\phi$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b[..., 0].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_r')

        ax = axs[1, 1]
        im = ax.imshow(b[..., 1].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_t')

        ax = axs[2, 1]
        im = ax.imshow(b[..., 2].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_p')

        fig.tight_layout()

        wandb.log({f"{log_prefix}Spherical": wandb.Image(fig)})
        plt.close('all')

        b_cartesian = np.einsum('...ij,...j->...i', transform, b) if transform is not None else b

        fig, axs = plt.subplots(3, 2, figsize=(8, 8))

        ax = axs[0, 0]
        im = ax.imshow(coords[..., 0].transpose(), origin='lower')
        ax.set_title('$x$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(coords[..., 1].transpose(), origin='lower')
        ax.set_title('$y$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[2, 0]
        im = ax.imshow(coords[..., 2].transpose(), origin='lower')
        ax.set_title('$z$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b_cartesian[..., 0].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_x')

        ax = axs[1, 1]
        im = ax.imshow(b_cartesian[..., 1].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_y')

        ax = axs[2, 1]
        im = ax.imshow(b_cartesian[..., 2].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('B_z')

        fig.tight_layout()

        wandb.log({f"{log_prefix}Cartesian": wandb.Image(fig)})
        plt.close('all')


class SphericalMapDataset(SphericalSliceDataset):

    def __init__(self, files, mask_configs=None, insert=None, **kwargs):
        if 'mu_filter' in kwargs:
            raise TypeError("mu_filter is no longer a SphericalMapDataset argument; use mask_configs instead.")

        # load maps
        r_map = _map(files['Br'])
        t_map = _map(files['Bt'])
        p_map = _map(files['Bp'])

        # load coordinates
        spherical_coords = _carrington_coordinates_from_map(r_map)
        r = spherical_coords.radius
        r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
        spherical_coords = np.stack([
            r.to_value(u.solRad),
            np.pi / 2 - spherical_coords.lat.to_value(u.rad),
            spherical_coords.lon.to_value(u.rad),
        ]).transpose()
        cartesian_coords = spherical_to_cartesian(spherical_coords)

        # load data and transform matrix
        b_spherical = np.stack([r_map.data, t_map.data, p_map.data]).transpose()
        nan_mask = ~np.isnan(b_spherical[..., 0]) & np.isnan(b_spherical[..., 1:]).any(-1)
        b_spherical[nan_mask, 1:] = 0  # set missing transverse components to zero
        b_cartesian = vector_spherical_to_cartesian(b_spherical, spherical_coords)
        transform = cartesian_to_spherical_matrix(spherical_coords)
        # load error maps
        if 'Br_err' in files and 'Bt_err' in files and 'Bp_err' in files:
            r_error_map = _map(files['Br_err'])
            t_error_map = _map(files['Bt_err'])
            p_error_map = _map(files['Bp_err'])
            b_error_spherical = np.stack([r_error_map.data,
                                          t_error_map.data,
                                          p_error_map.data]).transpose()
        else:
            b_error_spherical = None

        # insert additional data (e.g., HMI full disk maps)
        insert = insert if insert is not None else []
        insert = insert if isinstance(insert, list) else [insert]
        for insert_config in insert:
            insert_br = _map(insert_config['Br'])
            insert_bt = _map(insert_config['Bt'])
            insert_bp = _map(insert_config['Bp'])

            # reproject to reference map
            ref_map = _map(r_map.data, r_map.meta)
            # use reference time to avoid temporal shift
            ref_map.meta['date-obs'] = insert_br.date.to_datetime().isoformat()
            ref_wcs = ref_map.wcs
            insert_br = insert_br.reproject_to(ref_wcs)
            insert_bt = insert_bt.reproject_to(ref_wcs)
            insert_bp = insert_bp.reproject_to(ref_wcs)

            insert_b_spherical = np.stack([insert_br.data, insert_bt.data, insert_bp.data]).transpose()
            insert_b_cartesian = vector_spherical_to_cartesian(insert_b_spherical, spherical_coords)

            nan_mask = ~np.isnan(insert_b_spherical)
            b_spherical[nan_mask] = insert_b_spherical[nan_mask]
            b_cartesian[nan_mask] = insert_b_cartesian[nan_mask]

        # apply masking
        mask_configs = mask_configs if mask_configs is not None else []
        mask_configs = mask_configs if isinstance(mask_configs, list) else [mask_configs]
        for mask_config in mask_configs:
            self._mask(b_cartesian, b_error_spherical, b_spherical, cartesian_coords, mask_config, r_map,
                       spherical_coords,
                       transform)
        if np.any(np.isnan(cartesian_coords)):
            # crop nan values
            nan_coords = np.argwhere(~np.isnan(cartesian_coords).any(-1))
            min_x = nan_coords[..., 0].min()
            max_x = nan_coords[..., 0].max()
            min_y = nan_coords[..., 1].min()
            max_y = nan_coords[..., 1].max()
            # truncate data
            cartesian_coords = cartesian_coords[min_x:max_x, min_y:max_y]
            spherical_coords = spherical_coords[min_x:max_x, min_y:max_y]
            b_spherical = b_spherical[min_x:max_x, min_y:max_y]
            b_cartesian = b_cartesian[min_x:max_x, min_y:max_y]
            if b_error_spherical is not None:
                b_error_spherical = b_error_spherical[min_x:max_x, min_y:max_y]
            if transform is not None:
                transform = transform[min_x:max_x, min_y:max_y]

        super().__init__(b=b_spherical, coords=cartesian_coords, spherical_coords=spherical_coords,
                         b_err=b_error_spherical, transform=transform,
                         **kwargs)

    def _mask(self, b_cartesian, b_error_spherical, b_spherical, cartesian_coords, mask_config, r_map,
              spherical_coords, transform):
        m_type = mask_config['type']
        if m_type == 'mu_filter':
            mask = _mu_filter_mask(r_map, mask_config)
            mask = mask.T if mask is not None else np.zeros(spherical_coords.shape[:2], dtype=bool)
        elif m_type == 'reference':
            ref_map = _map(mask_config['file'])
            mask = self._reference_pixel_mask(ref_map, r_map, spherical_coords.shape[:2],
                                              mask_config.get('mu_filter'))
        elif m_type == 'helioprojective':
            ref_map = _map(mask_config['file'])
            with _suppress_sunpy_metadata_messages():
                reference_frame = ref_map.coordinate_frame
                bottom_left = SkyCoord(mask_config['Tx'][0] * u.arcsec, mask_config['Ty'][0] * u.arcsec,
                                       frame=reference_frame)
                top_right = SkyCoord(mask_config['Tx'][1] * u.arcsec, mask_config['Ty'][1] * u.arcsec,
                                     frame=reference_frame)
                bottom_left = bottom_left.transform_to(frames.HeliographicCarrington)
                top_right = top_right.transform_to(frames.HeliographicCarrington)
            slice_lon = np.array([bottom_left.lon.to(u.rad).value, top_right.lon.to(u.rad).value])
            slice_lat = np.array([bottom_left.lat.to(u.rad).value, top_right.lat.to(u.rad).value])
            slice_lat = sorted(np.pi / 2 - slice_lat)  # convert to spherical coordinates

            mask = (spherical_coords[..., 1] > slice_lat[0]) & \
                   (spherical_coords[..., 1] < slice_lat[1]) & \
                   (spherical_coords[..., 2] > slice_lon[0]) & \
                   (spherical_coords[..., 2] < slice_lon[1])
        elif m_type == 'heliographic_carrington':
            unit = mask_config['unit'] if 'unit' in mask_config else 'deg'
            slice_lon = u.Quantity(mask_config['longitude_range'], unit).to_value(u.rad)
            slice_lat = u.Quantity(mask_config['latitude_range'], unit).to_value(u.rad)
            slice_lat = sorted(np.pi / 2 - slice_lat)  # convert to spherical coordinates

            lat_mask = (spherical_coords[..., 1] > slice_lat[0]) & \
                       (spherical_coords[..., 1] < slice_lat[1])
            lon_mask = (spherical_coords[..., 2] > slice_lon[0]) & \
                       (spherical_coords[..., 2] < slice_lon[1])
            if slice_lon[1] > 2 * np.pi:
                lon_mask = lon_mask | \
                           ((spherical_coords[..., 2] > 0) & (spherical_coords[..., 2] < slice_lon[1] - 2 * np.pi))

            mask = lat_mask & lon_mask
        else:
            raise NotImplementedError(f"Unknown mask type '{m_type}'. "
                                      f"Please choose from: ['mu_filter', 'reference', 'helioprojective', "
                                      f"'heliographic_carrington']")

        mask = ~mask if 'invert' in mask_config and mask_config['invert'] else mask

        b_cartesian[mask] = np.nan
        b_spherical[mask] = np.nan
        if b_error_spherical is not None:
            b_error_spherical[mask] = np.nan
        cartesian_coords[mask] = np.nan
        spherical_coords[mask] = np.nan
        transform[mask] = np.nan

    @staticmethod
    def _reference_pixel_mask(ref_map, target_map, target_shape, mu_filter=None):
        ref_coords = _carrington_coordinates_from_map(ref_map)
        ref_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=ref_coords.radius,
                              frame=frames.HeliographicCarrington, obstime=target_map.date, observer='self')
        x_pix, y_pix = target_map.world_to_pixel(ref_coords)
        x_pix = x_pix.to_value(u.pix)
        y_pix = y_pix.to_value(u.pix)

        ny, nx = target_map.data.shape
        valid = np.isfinite(ref_map.data) & np.isfinite(x_pix) & np.isfinite(y_pix)
        mu_mask = _mu_filter_mask(ref_map, mu_filter)
        if mu_mask is not None:
            valid &= ~mu_mask
        x_idx = np.rint(x_pix[valid]).astype(int)
        y_idx = np.rint(y_pix[valid]).astype(int)
        in_bounds = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)

        mask = np.zeros(target_shape, dtype=bool)
        mask[x_idx[in_bounds], y_idx[in_bounds]] = True
        return mask

    @staticmethod
    def _mu_filter_mask(smap, mu_filter):
        return _mu_filter_mask(smap, mu_filter)


class SphericalFITSReferenceDataset(NF2Dataset):

    def __init__(self, Mm_per_ds, reference_br=None, reference_file=None, radius_range=None, batch_size=1024,
                 n_slices=10, requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        reference_br = reference_file if reference_br is None else reference_br
        if reference_br is None:
            raise ValueError('SphericalFITSReferenceDataset requires reference_br or reference_file.')
        reference_map = _map(reference_br)

        longitude_deg, latitude_deg = _reference_map_lon_lat(reference_map)
        self.reference_br = np.asarray(reference_map.data, dtype=np.float32)
        self.reference_lon = np.asarray(longitude_deg, dtype=np.float32)
        self.reference_lat = np.asarray(latitude_deg, dtype=np.float32)
        self.longitude_rad = np.deg2rad(self.reference_lon).astype(np.float32, copy=False)
        self.colatitude_rad = (np.pi / 2 - np.deg2rad(self.reference_lat)).astype(np.float32, copy=False)
        self.radius = np.linspace(radius_range[0], radius_range[1], int(n_slices), dtype=np.float32)
        self.radius_range = np.array([self.radius[0], self.radius[-1]], dtype=np.float32)
        self.cube_shape = (len(self.radius), *self.reference_br.shape)
        self.batch_size = int(batch_size)
        self.n_coords = int(np.prod(self.cube_shape))
        self.Mm_per_ds = Mm_per_ds
        self.coord_scale = spherical_coord_scale(Mm_per_ds)

    def __len__(self):
        return int(np.ceil(self.n_coords / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min((idx + 1) * self.batch_size, self.n_coords)
        flat_idx = np.arange(start, stop, dtype=np.int64)
        radius_idx, y_idx, x_idx = np.unravel_index(flat_idx, self.cube_shape)

        spherical_coords = np.stack([
            self.radius[radius_idx],
            self.colatitude_rad[y_idx, x_idx],
            self.longitude_rad[y_idx, x_idx],
        ], -1)
        coords = spherical_to_cartesian(spherical_coords) * self.coord_scale

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'spherical_coords': torch.tensor(spherical_coords, dtype=torch.float32),
            'reference_br': torch.tensor(self.reference_br[y_idx, x_idx, None], dtype=torch.float32),
            'reference_lon': torch.tensor(self.reference_lon[y_idx, x_idx, None], dtype=torch.float32),
            'reference_lat': torch.tensor(self.reference_lat[y_idx, x_idx, None], dtype=torch.float32),
        }

