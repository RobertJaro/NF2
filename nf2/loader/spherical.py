import datetime
import os
import uuid
from collections import OrderedDict
from copy import copy
from itertools import repeat

import numpy as np
import pfsspy
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BatchesDataset, RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset
from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian, cartesian_to_spherical_matrix


class SliceDataset(BatchesDataset):

    def __init__(self, b, coords, spherical_coords, G_per_dB, work_directory, batch_size,
                 error=None, transform=None,
                 plot_overview=True, filter_nans=True, shuffle=True, strides=1,
                 ds_name=None, **kwargs):
        if plot_overview:
            self._plot(b, coords, spherical_coords)
        if strides > 1:
            b = b[::strides, ::strides]
            coords = coords[::strides, ::strides]
            error = error[::strides, ::strides] if error is not None else None
            transform = transform[::strides, ::strides] if transform is not None else None
        self.cube_shape = b.shape[:-1]
        # flatten data
        b = np.concatenate([b.reshape((-1, 3)) for b in b]).astype(np.float32)
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        if error is not None:
            error = np.concatenate([e.reshape((-1, 3)) for e in error]).astype(np.float32)
        if transform is not None:
            transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)

        # filter nan entries
        nan_mask = np.all(np.isnan(b), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0 and filter_nans:
            print(f'Filtering {nan_mask.sum()} nan entries')
            b = b[~nan_mask]
            coords = coords[~nan_mask]
            transform = transform[~nan_mask] if transform is not None else None
            error = error[~nan_mask] if error is not None else None

        # normalize data
        b /= G_per_dB
        error = error / G_per_dB if error is not None else None

        # shuffle data
        if shuffle:
            r = np.random.permutation(coords.shape[0])
            coords = coords[r]
            b = b[r]
            transform = transform[r] if transform is not None else None
            error = error[r] if error is not None else None

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        coords_npy_path = os.path.join(work_directory, f'{ds_name}_coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        b_npy_path = os.path.join(work_directory, f'{ds_name}_b_true.npy')
        np.save(b_npy_path, b.astype(np.float32))
        batches_path = {'coords': coords_npy_path,
                        'b_true': b_npy_path}

        if transform is not None:
            transform_npy_path = os.path.join(work_directory, f'{ds_name}_transform.npy')
            np.save(transform_npy_path, transform.astype(np.float32))
            batches_path['transform'] = transform_npy_path
        if error is not None:
            err_npy_path = os.path.join(work_directory, f'{ds_name}_error.npy')
            np.save(err_npy_path, error.astype(np.float32))
            batches_path['error'] = err_npy_path

        self.batches_path = batches_path
        super().__init__(batches_path, batch_size)

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

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]


class MapDataset(SliceDataset):

    def __init__(self, files, mask_configs=[], **kwargs):
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
            np.pi / 2 + spherical_coords.lat.to_value(u.rad),
            spherical_coords.lon.to_value(u.rad),
        ]).transpose()
        cartesian_coords = spherical_to_cartesian(spherical_coords)

        # load data and transform matrix
        b_spherical = np.stack([r_map.data, -t_map.data, p_map.data]).transpose()
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

        # apply masking
        mask_configs = mask_configs if isinstance(mask_configs, list) else [mask_configs]
        for mask_config in mask_configs:
            self._mask(b_cartesian, b_error_spherical, b_spherical, cartesian_coords, mask_config, r_map,
                       spherical_coords,
                       transform)
        if np.any(np.isnan(cartesian_coords)):
            # crop nan values
            nan_coords = np.argwhere(~np.isnan(cartesian_coords[:, :]).any(-1))
            min_x = nan_coords[..., 0].min()
            max_x = nan_coords[..., 0].max()
            min_y = nan_coords[..., 1].min()
            max_y = nan_coords[..., 1].max()
            # truncate data
            cartesian_coords = cartesian_coords[min_x:max_x, min_y:max_y]
            spherical_coords = spherical_coords[min_x:max_x, min_y:max_y]
            b_spherical = b_spherical[min_x:max_x, min_y:max_y]
            if b_error_spherical is not None:
                b_error_spherical = b_error_spherical[min_x:max_x, min_y:max_y]
            transform = transform[min_x:max_x, min_y:max_y] if transform is not None else None

        super().__init__(b=b_spherical, coords=cartesian_coords, spherical_coords=spherical_coords,
                         error=b_error_spherical, transform=transform,
                         **kwargs)

    def _mask(self, b_cartesian, b_error_spherical, b_spherical, cartesian_coords, mask_config, r_map,
              spherical_coords, transform):
        m_type = mask_config['type']
        if m_type == 'reference':
            ref_map = Map(mask_config['file'])
            ref_map.meta['date-obs'] = r_map.date.to_datetime().isoformat() # use reference time to avoid temporal shift
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
            slice_lon = mask_config['longitude_range']
            slice_lat = mask_config['latitude_range']

            mask = (spherical_coords[..., 1] > slice_lat[0]) & \
                   (spherical_coords[..., 1] < slice_lat[1]) & \
                   (spherical_coords[..., 2] > slice_lon[0]) & \
                   (spherical_coords[..., 2] < slice_lon[1])
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




class PFSSBoundaryDataset(SliceDataset):

    def __init__(self, Br, height, source_surface_height=2.5, resample=[360, 180], sampling_points=100, **kwargs):
        assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

        # PFSS extrapolation
        potential_r_map = Map(Br)
        potential_r_map = potential_r_map.resample(resample * u.pix)
        pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
        pfss_out = pfsspy.pfss(pfss_in)

        # load B field
        ref_coords = all_coordinates_from_map(potential_r_map)
        spherical_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=height * u.solRad,
                                    frame=ref_coords.frame)
        potential_shape = spherical_coords.shape  # required workaround for pfsspy spherical reshape
        spherical_b = pfss_out.get_bvec(spherical_coords.reshape((-1,)))
        spherical_b = spherical_b.reshape((*potential_shape, 3)).value
        spherical_b[..., 1] *= -1  # flip B_theta
        spherical_b = np.stack([spherical_b[..., 0],
                                spherical_b[..., 1],
                                spherical_b[..., 2]]).T

        # load coordinates
        spherical_coords = np.stack([
            spherical_coords.radius.value,
            np.pi / 2 + spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value]).T

        # convert to spherical coordinates
        b = vector_spherical_to_cartesian(spherical_b, spherical_coords)
        coords = spherical_to_cartesian(spherical_coords)
        transform = cartesian_to_spherical_matrix(spherical_coords)

        super().__init__(b=b, coords=coords, spherical_coords=spherical_coords, transform=transform, **kwargs)


class SphericalDataModule(LightningDataModule):

    def __init__(self, train_configs, validation_configs,
                 max_radius=None, G_per_dB=None, work_directory=None,
                 batch_size=4096, num_workers=None,
                 **kwargs):
        super().__init__()

        # data parameters
        self.max_radius = max_radius
        self.G_per_dB = G_per_dB
        self.cube_shape = [1, max_radius]
        self.spatial_norm = 1 * u.solRad
        # train parameters
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        # init boundary datasets
        datasets = {'train': OrderedDict(),
                    'validation': OrderedDict()}

        iter_config = list(zip(train_configs, repeat('train'))) + list(zip(validation_configs, repeat('validation')))
        for i, (config, c_set) in enumerate(iter_config):
            c_type = config.pop('type')
            c_name = config.pop('name') if 'name' in config else f'dataset_{c_type}_{i}'
            config['work_directory'] = work_directory if 'work_directory' not in config else config['work_directory']
            os.makedirs(config['work_directory'], exist_ok=True)
            config['batch_size'] = batch_size if 'batch_size' not in config else config['batch_size']
            config['G_per_dB'] = G_per_dB if 'G_per_dB' not in config else config['G_per_dB']
            config['ds_name'] = c_name
            if c_type == 'map':
                dataset = MapDataset(**config)
            elif c_type == 'pfss_boundary':
                config['height'] = max_radius if 'height' not in config else config['height']
                dataset = PFSSBoundaryDataset(**config)
            elif c_type == 'random_spherical':
                config['radius_range'] = [1, max_radius] if 'radius_range' not in config else config['radius_range']
                dataset = RandomSphericalCoordinateDataset(**config)
            elif c_type == 'sphere':
                config['radius_range'] = [1, max_radius] if 'radius_range' not in config else config['radius_range']
                dataset = SphereDataset(**config)
            elif c_type == 'spherical_slices':
                config['radius_range'] = [1, max_radius] if 'radius_range' not in config else config['radius_range']
                dataset = SphereSlicesDataset(**config)
            else:
                raise NotImplementedError(f"Unknown data type '{c_type}'. "
                                          f"Please choose from: ['map', 'potential_boundary', 'random_spherical', 'sphere', 'slices']")
            datasets[c_set][c_name] = dataset

        # create data loaders
        self.datasets = datasets
        self.validation_dataset_mapping = {i: name for i, name in enumerate(datasets['validation'].keys())}
        self.config = {'type': 'spherical',
                       'radius_range': [1, max_radius] * u.solRad,
                       'G_per_dB': G_per_dB * u.G,
                       'Mm_per_ds': (1 * u.solRad).to(u.Mm)}

    def clear(self):
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, SliceDataset)]

    def train_dataloader(self):
        datasets = self.datasets['train']
        ref_idx = np.argmax([len(ds) for ds in datasets.values()])
        ref_dataset_name, ref_dataset = list(datasets.items())[ref_idx]
        loaders = {ref_dataset_name: DataLoader(ref_dataset, batch_size=None, num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)
                   }
        for i, (name, dataset) in enumerate(datasets.items()):
            if i == ref_idx:
                continue # reference dataset already added
            sampler = RandomSampler(dataset, replacement=True, num_samples=len(ref_dataset))
            loaders[name] = DataLoader(dataset, batch_size=None, num_workers=self.num_workers,
                                       pin_memory=True, sampler=sampler)
        return loaders

    def val_dataloader(self):
        datasets = self.datasets['validation']
        loaders = []
        for dataset in datasets.values():
            loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                shuffle=False)
            loaders.append(loader)
        return loaders


class SphericalSeriesDataModule(SphericalDataModule):
    def __init__(self, full_disk_files, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.full_disk_files = copy(full_disk_files)
        self.current_files = self.full_disk_files[0]

        super().__init__(full_disk_files=self.full_disk_files[0], *self.args, **self.kwargs)

    def train_dataloader(self):
        if len(self.full_disk_files) == 0:
            return None
        # re-initialize
        print(f"Load next file: {os.path.basename(self.full_disk_files[0]['Br'])}")
        self.current_files = self.full_disk_files[0]
        super().__init__(full_disk_files=self.full_disk_files[0], *self.args, **self.kwargs)
        del self.full_disk_files[0]
        return super().train_dataloader()
