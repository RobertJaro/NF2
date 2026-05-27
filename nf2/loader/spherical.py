import os
import glob
import re
import multiprocessing as mp
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import rank_zero_only
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.dataset import RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset, \
    RandomRadialGroupedCoordinateDataset, TensorsDataset
from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian, cartesian_to_spherical_matrix
from nf2.loader.base import BaseDataModule


def _mu_filter_mask(smap, mu_filter):
    if mu_filter is None:
        return None
    if isinstance(mu_filter, (int, float)):
        mu_filter = {'min': mu_filter}

    coords = all_coordinates_from_map(smap)
    rho = np.hypot(coords.Tx.to_value(u.rad), coords.Ty.to_value(u.rad))
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
    carrington_coords = all_coordinates_from_map(reference_map).transform_to(frames.HeliographicCarrington)
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
        coords = coords * (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

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
        r_map = Map(files['Br'])
        t_map = Map(files['Bt'])
        p_map = Map(files['Bp'])

        # load coordinates
        spherical_coords = all_coordinates_from_map(r_map).transform_to(frames.HeliographicCarrington)
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
            r_error_map = Map(files['Br_err'])
            t_error_map = Map(files['Bt_err'])
            p_error_map = Map(files['Bp_err'])
            b_error_spherical = np.stack([r_error_map.data,
                                          t_error_map.data,
                                          p_error_map.data]).transpose()
        else:
            b_error_spherical = None

        # insert additional data (e.g., HMI full disk maps)
        insert = insert if insert is not None else []
        insert = insert if isinstance(insert, list) else [insert]
        for insert_config in insert:
            insert_br = Map(insert_config['Br'])
            insert_bt = Map(insert_config['Bt'])
            insert_bp = Map(insert_config['Bp'])

            # reproject to reference map
            ref_map = Map(r_map.data, r_map.meta)
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
            ref_map = Map(mask_config['file'])
            mask = self._reference_pixel_mask(ref_map, r_map, spherical_coords.shape[:2],
                                              mask_config.get('mu_filter'))
        elif m_type == 'helioprojective':
            ref_map = Map(mask_config['file'])
            bottom_left = SkyCoord(mask_config['Tx'][0] * u.arcsec, mask_config['Ty'][0] * u.arcsec,
                                   frame=ref_map.coordinate_frame)
            top_right = SkyCoord(mask_config['Tx'][1] * u.arcsec, mask_config['Ty'][1] * u.arcsec,
                                 frame=ref_map.coordinate_frame)
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
        ref_coords = all_coordinates_from_map(ref_map).transform_to(frames.HeliographicCarrington)
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


class SphericalFITSReferenceDataset(TensorsDataset):

    def __init__(self, Mm_per_ds, reference_br=None, reference_file=None, radius_range=None, batch_size=1024,
                 n_slices=10, **kwargs):
        reference_br = reference_file if reference_br is None else reference_br
        if reference_br is None:
            raise ValueError('SphericalFITSReferenceDataset requires reference_br or reference_file.')
        reference_map = Map(reference_br)

        longitude_deg, latitude_deg = _reference_map_lon_lat(reference_map)
        longitude_rad = np.deg2rad(longitude_deg)
        colatitude_rad = np.pi / 2 - np.deg2rad(latitude_deg)
        radius = np.linspace(radius_range[0], radius_range[1], int(n_slices), dtype=np.float32)

        spherical_coords = np.empty((len(radius), *reference_map.data.shape, 3), dtype=np.float32)
        spherical_coords[..., 0] = radius[:, None, None]
        spherical_coords[..., 1] = colatitude_rad[None]
        spherical_coords[..., 2] = longitude_rad[None]

        cartesian_coords = spherical_to_cartesian(spherical_coords)
        cartesian_coords = cartesian_coords * (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

        self.cube_shape = spherical_coords.shape[:-1]
        tensors = {
            'coords': cartesian_coords.reshape(-1, 3),
            'spherical_coords': spherical_coords.reshape(-1, 3),
            'reference_br': np.broadcast_to(reference_map.data[None], self.cube_shape).reshape(-1, 1),
            'reference_lon': np.broadcast_to(longitude_deg[None], self.cube_shape).reshape(-1, 1),
            'reference_lat': np.broadcast_to(latitude_deg[None], self.cube_shape).reshape(-1, 1),
        }

        super().__init__(tensors, batch_size=batch_size, Mm_per_ds=Mm_per_ds, **kwargs)


class SphericalDataModule(BaseDataModule):

    def __init__(self, boundaries, validation, samplers=None,
                 max_radius=1.3,
                 Mm_per_ds=100,
                 Gauss_per_dB=1000, work_path=None,
                 batch_size=4096, type=None, geometry=None, **kwargs):

        self.ds_mapping = {'map': SphericalMapDataset,
                           'fits_reference': SphericalFITSReferenceDataset,
                           'random_spherical': RandomSphericalCoordinateDataset,
                           'random_radial_grouped': RandomRadialGroupedCoordinateDataset,
                           'sphere': SphereDataset,
                           'spherical_slices': SphereSlicesDataset}

        # data parameters
        self.Gauss_per_dB = Gauss_per_dB
        self.Mm_per_ds = Mm_per_ds
        self.cube_shape = [1, max_radius]
        self.spatial_norm = 1 * u.solRad

        # init boundary datasets
        general_config = {'work_path': work_path, 'batch_size': batch_size, 'Gauss_per_dB': Gauss_per_dB,
                          'radius_range': [1, max_radius], 'Mm_per_ds': Mm_per_ds}

        config = {'schema_version': '0.4',
                  'type': 'spherical',
                  'geometry': 'spherical',
                  'coordinate_system': 'heliographic_carrington',
                  'boundary_field_components': ['Br', 'Btheta', 'Bphi'],
                  'model_field_components': ['Bx', 'By', 'Bz'],
                  'field_unit': 'G',
                  'length_unit': 'Mm',
                  'radius_unit': 'solRad',
                  'max_radius': max_radius,
                  'radius_range': [1, max_radius],
                  'spatial_norm_Mm': (1 * u.solRad).to_value(u.Mm),
                  'Gauss_per_dB': Gauss_per_dB,
                  'Mm_per_ds': Mm_per_ds,
                  'normalization': {'Mm_per_ds': Mm_per_ds, 'Gauss_per_dB': Gauss_per_dB}}

        training_configs = list(boundaries) + list(samplers or [])
        training_datasets = self.load_config(training_configs, general_config, prefix='train')
        validation_datasets = self.load_config(validation, general_config, prefix='validation')

        super().__init__(training_datasets, validation_datasets, config, **kwargs)

    def load_config(self, configs, general_config, prefix='train'):
        datasets = OrderedDict()
        for i, config in enumerate(configs):
            config = deepcopy(config)
            c_type = config.pop('type')
            c_name = config.pop('id') if 'id' in config else f'{prefix}_{c_type}_{i}'
            requires_jacobian = config.pop('requires_jacobian', True)
            config['ds_name'] = c_name
            # update config with general config
            for k, v in general_config.items():
                if k not in config:
                    config[k] = v
            if c_type in ['random_spherical', 'random_radial_grouped']:
                self._apply_random_reference_map(config)
            config['requires_jacobian'] = requires_jacobian
            os.makedirs(config['work_path'], exist_ok=True)
            dataset = self.ds_mapping[c_type](**config)
            dataset.config = {
                'id': c_name,
                'type': c_type,
                'role': prefix,
                **deepcopy(config),
            }
            datasets[c_name] = dataset
        return datasets

    @staticmethod
    def _apply_random_reference_map(config):
        reference_config = config.pop('reference_map', None)
        if reference_config is None:
            return

        if isinstance(reference_config, str):
            reference_config = {'file': reference_config}
        if not isinstance(reference_config, dict):
            raise TypeError('reference_map must be a file path or configuration dictionary.')

        reference_file = reference_config.get('file')
        if reference_file is None:
            raise ValueError('reference_map requires a file.')

        reference_map = Map(reference_file)
        mu_filter = reference_config.get('mu_filter', reference_config.get('mu'))
        bounds = _reference_map_coordinate_bounds(reference_map, mu_filter, f'reference_map file: {reference_file}')

        config.setdefault('latitude_range', bounds['latitude_range'])
        config.setdefault('longitude_range', bounds['longitude_range'])
        config.setdefault('unit', 'deg')


def _load_spherical_data_module(worker_args):
    step, total_steps, boundaries, args, kwargs = worker_args
    print(f'Loading data module {step + 1:03d}/{total_steps:03d}; '
          f'ID: {SphericalSeriesDataModule._step_id(boundaries, step)}')
    return SphericalDataModule(boundaries=boundaries, *args, **kwargs)


class SphericalSeriesDataModule(LightningDataModule):

    def __init__(self, boundaries, samplers=None, current_step=0, iterations=None, data_module_workers=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.iterations = iterations

        self.boundaries = self._expand_boundaries(list(boundaries) + list(samplers or []))
        self.step = current_step
        self.total_steps = len(self.boundaries)

        assert self.step < self.total_steps, \
            'Not enough data files found to continue training. Training completed or configuration is incorrect.'

        self.data_modules = [None] * self.total_steps
        super().__init__()
        self._load_data_modules(data_module_workers)

    def _expand_boundaries(self, boundaries):
        expanded_configs = [self._expand_config(config) for config in boundaries]
        total_steps = max(len(configs) for configs in expanded_configs)
        assert all(len(configs) in (1, total_steps) for configs in expanded_configs), \
            'Inconsistent number of training files in configurations. Check your configurations.'

        configs_by_step = []
        for step in range(total_steps):
            step_configs = [deepcopy(configs[step] if len(configs) > 1 else configs[0])
                            for configs in expanded_configs]
            context = {config.get('id', f'train_{i}'): config
                       for i, config in enumerate(step_configs)}
            configs_by_step.append([self._resolve_placeholders(config, context) for config in step_configs])
        return configs_by_step

    def _expand_config(self, config):
        config = deepcopy(config)
        if config['type'] == 'map':
            configs = self._expand_map_config(config)
        else:
            configs = [config]

        if self.iterations is not None:
            for c in configs:
                if c['type'] in ['random_spherical', 'random_radial_grouped'] and 'length' not in c:
                    c['length'] = self.iterations
        return configs

    def _expand_map_config(self, config):
        files = config.get('files')
        if files is None:
            return [config]
        if isinstance(files, list):
            configs = []
            for f in files:
                c = deepcopy(config)
                c['files'] = f
                c['id'] = self._files_id(f)
                configs.append(c)
            return configs

        expanded_files = {k: self._expand_file_value(v) for k, v in files.items()}
        series_lengths = [len(v) for v in expanded_files.values() if isinstance(v, list)]
        if len(series_lengths) == 0:
            return [config]

        n_steps = max(series_lengths)
        assert all(length in (1, n_steps) for length in series_lengths), \
            f'Inconsistent number of files in spherical map config {config.get("id", "")}'

        configs = []
        for step in range(n_steps):
            c = deepcopy(config)
            c['files'] = {k: v[step if len(v) > 1 else 0] if isinstance(v, list) else v
                          for k, v in expanded_files.items()}
            c['id'] = self._files_id(c['files'])
            configs.append(c)
        return configs

    @staticmethod
    def _expand_file_value(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and glob.has_magic(value):
            files = sorted(glob.glob(value))
            assert len(files) > 0, f'No files found for pattern {value}'
            return files
        return value

    @staticmethod
    def _resolve_placeholders(value, context):
        if isinstance(value, dict):
            return {k: SphericalSeriesDataModule._resolve_placeholders(v, context)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [SphericalSeriesDataModule._resolve_placeholders(v, context) for v in value]
        if not isinstance(value, str):
            return value

        def replace(match):
            keys = match.group(1).split('.')
            resolved = context[keys[0]]
            for key in keys[1:]:
                resolved = resolved[key]
            return str(resolved)

        return re.sub(r'<<([^<>]+)>>', replace, value)

    @staticmethod
    def _files_id(files):
        if 'Br' in files:
            return os.path.basename(files['Br']).split('.')[-3]
        return None

    @staticmethod
    def _step_id(boundaries, step):
        for config in boundaries:
            if 'id' in config and config['id'] is not None:
                return config['id']
            if config.get('type') == 'map' and 'files' in config:
                config_id = SphericalSeriesDataModule._files_id(config['files'])
                if config_id is not None:
                    return config_id
        return f'step_{step:06d}'

    @property
    def current_id(self):
        return self._step_id(self.boundaries[self.step], self.step)

    def _get_data_module(self, step):
        return self.data_modules[step]

    def _load_data_modules(self, data_module_workers):
        n_workers = data_module_workers if data_module_workers is not None else (os.cpu_count() or 1)
        n_workers = max(1, min(n_workers, self.total_steps))

        print(f'Loading data modules... (total: {self.total_steps}, workers: {n_workers})')
        worker_args = [(step, self.total_steps, boundaries, self.args, self.kwargs)
                       for step, boundaries in enumerate(self.boundaries)]
        if n_workers == 1:
            self.data_modules = [_load_spherical_data_module(args) for args in worker_args]
            return

        start_method = 'fork' if 'fork' in mp.get_all_start_methods() else None
        context = mp.get_context(start_method) if start_method is not None else mp.get_context()
        with context.Pool(processes=n_workers) as pool:
            self.data_modules = pool.map(_load_spherical_data_module, worker_args)

    @property
    def config(self):
        return self._get_data_module(self.step).config

    @property
    def validation_datasets(self):
        return self._get_data_module(self.step).validation_datasets

    @property
    def validation_dataset_mapping(self):
        return self._get_data_module(self.step).validation_dataset_mapping

    def train_dataloader(self):
        print('Currently loaded:', self.current_id)
        return self._get_data_module(self.step).train_dataloader()

    def val_dataloader(self):
        return self._get_data_module(self.step).val_dataloader()
