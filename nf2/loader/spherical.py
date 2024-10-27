import os
from collections import OrderedDict
from copy import copy, deepcopy

import numpy as np
import pfsspy
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.dataset import RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset
from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian, cartesian_to_spherical_matrix
from nf2.loader.base import BaseDataModule, TensorsDataset


class SphericalSliceDataset(TensorsDataset):

    def __init__(self, b, coords, spherical_coords, G_per_dB, Mm_per_ds,
                 b_err=None, transform=None,
                 plot_overview=True, strides=1, **kwargs):
        if plot_overview:
            self._plot(b, coords, spherical_coords)
        if strides > 1:
            b = b[::strides, ::strides]
            coords = coords[::strides, ::strides]
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
        b /= G_per_dB
        b_err = b_err / G_per_dB if b_err is not None else None
        coords = coords * (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

        tensors = {'coords': coords,
                   'b_true': b, }
        if transform is not None:
            tensors['transform'] = transform
        if b_err is not None:
            tensors['b_err'] = b_err

        super().__init__(tensors, **kwargs)

    def _plot(self, b, coords, spherical_coords):
        b_min_max = np.nanmax(np.abs(b))
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        im = axs[0].imshow(b[..., 0].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        im = axs[1].imshow(b[..., 1].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        im = axs[2].imshow(b[..., 2].transpose(), vmin=-b_min_max, vmax=b_min_max, cmap='gray', origin='lower')
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        wandb.log({"Overview - B": fig})
        plt.close('all')

        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        im = axs[0].imshow(coords[..., 0].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(coords[..., 1].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(coords[..., 2].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[2])
        wandb.log({"Coordinates": fig})
        plt.close('all')

        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        im = axs[0].imshow(spherical_coords[..., 0].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(spherical_coords[..., 1].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow(spherical_coords[..., 2].transpose(), origin='lower')
        fig.colorbar(im, ax=axs[2])
        wandb.log({"Spherical Coordinates": fig})
        plt.close('all')


class SphericalMapDataset(SphericalSliceDataset):

    def __init__(self, files, mask_configs=None, insert=None, **kwargs):
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
        if m_type == 'reference':
            ref_map = Map(mask_config['file'])
            ref_map.meta[
                'date-obs'] = r_map.date.to_datetime().isoformat()  # use reference time to avoid temporal shift
            reprojected_map = ref_map.reproject_to(r_map.wcs)
            mask = ~np.isnan(reprojected_map.data).T
        elif m_type == 'helioprojective':
            ref_map = Map(mask_config['file'])
            bottom_left = SkyCoord(mask_config['Tx'][0] * u.arcsec, mask_config['Ty'][0] * u.arcsec,
                                   frame=ref_map.coordinate_frame)
            top_right = SkyCoord(mask_config['Tx'][1] * u.arcsec, mask_config['Ty'][1] * u.arcsec,
                                 frame=ref_map.coordinate_frame)
            bottom_left = bottom_left.transform_to(frames.HeliographicCarrington)
            top_right = top_right.transform_to(frames.HeliographicCarrington)
            slice_lon = np.array([bottom_left.lon.to(u.rad).value, top_right.lon.to(u.rad).value])
            slice_lat = np.array([bottom_left.lat.to(u.rad).value, top_right.lat.to(u.rad).value]) + np.pi / 2

            mask = (spherical_coords[..., 1] > slice_lat[0]) & \
                   (spherical_coords[..., 1] < slice_lat[1]) & \
                   (spherical_coords[..., 2] > slice_lon[0]) & \
                   (spherical_coords[..., 2] < slice_lon[1])
        elif m_type == 'heliographic_carrington':
            unit = u.Quantity(mask_config['unit']) if 'unit' in mask_config else u.rad
            slice_lon = (mask_config['longitude_range'] * unit).to_value(u.rad)
            slice_lat = (mask_config['latitude_range'] * unit).to_value(u.rad)

            lat_mask = (spherical_coords[..., 1] > slice_lat[0]) & \
                       (spherical_coords[..., 1] < slice_lat[1])
            lon_mask = (spherical_coords[..., 2] > slice_lon[0]) & \
                       (spherical_coords[..., 2] < slice_lon[1])
            if slice_lon[1] > 2 * np.pi:
                lon_mask = lon_mask | \
                           ((spherical_coords[..., 2] > 0) & (spherical_coords[..., 2] < slice_lon[1] - 2 * np.pi))

            mask = lat_mask & lon_mask
        else:
            raise NotImplementedError(f"Unknown mask type '{type}'. "
                                      f"Please choose from: ['reference', 'helioprojective', 'heliographic_carrington']")

        mask = ~mask if 'invert' in mask_config and mask_config['invert'] else mask

        b_cartesian[mask] = np.nan
        b_spherical[mask] = np.nan
        if b_error_spherical is not None:
            b_error_spherical[mask] = np.nan
        cartesian_coords[mask] = np.nan
        spherical_coords[mask] = np.nan
        transform[mask] = np.nan


class PFSSBoundaryDataset(SphericalSliceDataset):

    def __init__(self, Br, radius_range, source_surface_height=2.5, resample=[360, 180], sampling_points=100, mask=None,
                 insert=None, **kwargs):
        height = radius_range[1]
        assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

        # load synoptic map
        potential_r_map = Map(Br)
        potential_r_map = potential_r_map.resample(resample * u.pix)

        # insert additional data (e.g., HMI full disk maps)
        insert = insert if insert is not None else []
        insert = insert if isinstance(insert, list) else [insert]
        for file_in in insert:
            map_in = Map(file_in)
            # prevent temporal rotation
            potential_r_map.meta['date-obs'] = map_in.date.to_datetime().isoformat()
            map_in = map_in.reproject_to(potential_r_map.wcs)
            condition = ~np.isnan(map_in.data)
            potential_r_map.data[condition] = map_in.data[condition]

        # PFSS extrapolation
        pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
        pfss_out = pfsspy.pfss(pfss_in)

        # load B field
        ref_coords = all_coordinates_from_map(potential_r_map)
        spherical_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=height * u.solRad,
                                    frame=ref_coords.frame)
        potential_shape = spherical_coords.shape  # required workaround for pfsspy spherical reshape
        spherical_b = pfss_out.get_bvec(spherical_coords.reshape((-1,)))
        spherical_b = spherical_b.reshape((*potential_shape, 3)).value
        spherical_b = np.stack([spherical_b[..., 0],
                                spherical_b[..., 1],
                                spherical_b[..., 2]]).T

        # load coordinates
        spherical_coords = np.stack([
            spherical_coords.radius.value,
            np.pi / 2 - spherical_coords.lat.to_value(u.rad),
            spherical_coords.lon.to(u.rad).value]).T

        if mask is not None:
            condition = (spherical_coords[..., 1] < mask['latitude_range'][0]) | \
                        (spherical_coords[..., 1] > mask['latitude_range'][1]) | \
                        (spherical_coords[..., 2] < mask['longitude_range'][0]) | \
                        (spherical_coords[..., 2] > mask['longitude_range'][1])
            spherical_b[condition] = np.nan
            spherical_coords[condition] = np.nan

        # convert to spherical coordinates
        coords = spherical_to_cartesian(spherical_coords)
        transform = cartesian_to_spherical_matrix(spherical_coords)


        super().__init__(b=spherical_b, coords=coords, spherical_coords=spherical_coords, transform=transform, **kwargs)


class SphericalDataModule(BaseDataModule):

    def __init__(self, train_configs, validation_configs,
                 max_radius=1.3,
                 Mm_per_ds= (1 * u.solRad).to_value(u.Mm),
                 G_per_dB=None, work_directory=None,
                 batch_size=4096, **kwargs):

        self.ds_mapping = {'map': SphericalMapDataset,
                           'pfss_boundary': PFSSBoundaryDataset,
                           'random_spherical': RandomSphericalCoordinateDataset,
                           'sphere': SphereDataset,
                           'spherical_slices': SphereSlicesDataset}

        # data parameters
        self.G_per_dB = G_per_dB
        self.Mm_per_ds = Mm_per_ds
        self.cube_shape = [1, max_radius]
        self.spatial_norm = 1 * u.solRad

        # init boundary datasets
        general_config = {'work_directory': work_directory, 'batch_size': batch_size, 'G_per_dB': G_per_dB,
                          'radius_range': [1, max_radius], 'Mm_per_ds': Mm_per_ds}

        config = {'type': 'spherical',
                  'radius_range': [1, max_radius],
                  'G_per_dB': G_per_dB,
                  'Mm_per_ds': Mm_per_ds}

        training_datasets = self.load_config(train_configs, general_config, prefix='train')
        validation_datasets = self.load_config(validation_configs, general_config, prefix='validation')

        super().__init__(training_datasets, validation_datasets, config, **kwargs)

    def load_config(self, configs, general_config, prefix='train'):
        datasets = OrderedDict()
        for i, config in enumerate(configs):
            config = deepcopy(config)
            c_type = config.pop('type')
            c_name = config.pop('ds_id') if 'ds_id' in config else f'{prefix}_{c_type}_{i}'
            config['ds_name'] = c_name
            # update config with general config
            for k, v in general_config.items():
                if k not in config:
                    config[k] = v
            os.makedirs(config['work_directory'], exist_ok=True)
            dataset = self.ds_mapping[c_type](**config)
            datasets[c_name] = dataset
        return datasets


class SphericalSeriesDataModule(SphericalDataModule):

    def __init__(self, fits_paths, synoptic_fits_path, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fits_paths = copy(fits_paths)
        self.synoptic_fits_path = synoptic_fits_path

        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]

        self.initialized = True  # only required for first iteration
        train_configs = self._build_config(fits_paths[0])
        super().__init__(train_configs, *self.args, **self.kwargs)

    def _build_config(self, fits):
        full_disk_config = {
            'type': 'map',
            'ds_id': 'full_disk',
            'batch_size': 4096,
            'files': {'Br': fits['Br'], 'Bt': fits['Bt'], 'Bp': fits['Bp'], }
        }
        synoptic_config = {
            'type': 'map',
            'ds_id': 'synoptic',
            'batch_size': 4096,
            'files': {'Br': self.synoptic_fits_path['Br'],
                      'Bt': self.synoptic_fits_path['Bt'],
                      'Bp': self.synoptic_fits_path['Bp']},
            'mask_configs': [{'type': 'reference', 'file': fits['Br']}]
        }
        random_config = {
            'type': 'random_spherical',
            'ds_id': 'random',
            'batch_size': 16384
        }
        train_configs = [full_disk_config, synoptic_config, random_config]
        return train_configs

    def train_dataloader(self):
        # skip reload if already initialized - for initial epoch
        if self.initialized:
            self.initialized = False
            print('Currently loaded:', self.current_id)
            return super().train_dataloader()
        # update ID
        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]
        # re-initialize
        train_configs = self._build_config(self.fits_paths[0])
        super().__init__(train_configs, *self.args, **self.kwargs)
        # continue with next file in list
        del self.fits_paths[0]
        print('Currently loaded:', self.current_id)
        return super().train_dataloader()
