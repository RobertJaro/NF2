import glob
import os
from copy import copy

import numpy as np
import pfsspy
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset, RandomSphericalCoordinateDataset, \
    SphereDataset, SphereSlicesDataset
from nf2.data.loader import prep_b_data, load_potential_field_data
from nf2.data.util import vector_spherical_to_cartesian, spherical_to_cartesian, cartesian_to_spherical_matrix, \
    cartesian_to_spherical
from nf2.train.model import image_to_spherical_matrix


class SlicesDataModule(LightningDataModule):

    def __init__(self, b_slices, height, spatial_norm, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 error_slices=None, height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_strides = 1,
                 meta_data=None, plot_overview=True, Mm_per_pixel=None, buffer=None,
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        self.Mm_per_pixel = Mm_per_pixel

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        self.b_slices = b_slices

        if plot_overview:
            for i in range(b_slices.shape[2]):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b_slices[..., i, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b_slices[..., i, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b_slices[..., i, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')

        # load dataset
        assert len(height_mapping['z']) == b_slices.shape[2], 'Invalid height mapping configuration: z must have the same length as the number of slices'
        coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        for i, h in enumerate(height_mapping['z']):
            coords[:, :, i, 2] = h
        ranges = np.zeros((*coords.shape[:-1], 2))
        use_height_range = 'z_max' in height_mapping
        if use_height_range:
            z1 = height_mapping['z_max']
            # set to lower boundary if not specified
            z0 = height_mapping['z_min'] if 'z_min' in height_mapping else np.zeros_like(z1)
            assert len(z0) == len(z1) == len(height_mapping['z']), \
                'Invalid height mapping configuration: z_min, z_max and z must have the same length'
            for i, (h_min, h_max) in enumerate(zip(z0, z1)):
                ranges[:, :, i, 0] = h_min
                ranges[:, :, i, 1] = h_max
        # flatten data
        coords = coords.reshape((-1, 3)).astype(np.float32)
        values = b_slices.reshape((-1, 3)).astype(np.float32)
        ranges = ranges.reshape((-1, 2)).astype(np.float32)
        errors = error_slices.reshape((-1, 3)).astype(np.float32) if error_slices is not None else np.zeros_like(values)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            values = values[~nan_mask]
            ranges = ranges[~nan_mask]
            errors = errors[~nan_mask]

        if boundary['type'] == 'potential':
            b_bottom = b_slices[:, :, 0]
            b_bottom = np.nan_to_num(b_bottom, nan=0) # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'], progress=True)
            #
            pf_ranges = np.zeros((*pf_coords.shape[:-1], 2), dtype=np.float32)
            pf_ranges[:, 0] = pf_coords[:, 2]
            pf_ranges[:, 1] = pf_coords[:, 2]
            # concatenate pf data points
            coords = np.concatenate([pf_coords, coords])
            values = np.concatenate([pf_values, values])
            ranges = np.concatenate([pf_ranges, ranges])
            errors = np.concatenate([pf_errors, errors])
        elif boundary['type'] == 'potential_top':
            b_bottom = b_slices[:, :, 0]
            b_bottom = np.nan_to_num(b_bottom, nan=0) # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'], only_top=True, pf_error=0.1, progress=True)
            # log upper boundary
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            pf_b_map = pf_values.reshape(b_bottom.shape)
            axs[0].imshow(pf_b_map[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[1].imshow(pf_b_map[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[2].imshow(pf_b_map[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')
            #
            pf_ranges = np.zeros((*pf_coords.shape[:-1], 2), dtype=np.float32)
            pf_ranges[:, 0] = height / 2
            pf_ranges[:, 1] = height
            # concatenate pf data points
            coords = np.concatenate([pf_coords, coords])
            values = np.concatenate([pf_values, values])
            ranges = np.concatenate([pf_ranges, ranges])
            errors = np.concatenate([pf_errors, errors])
        elif boundary['type'] == 'open':
            pass
        else:
            raise ValueError('Unknown boundary type')

        # normalize data
        values = values / b_norm
        errors = errors / b_norm
        # apply spatial normalization
        coords = coords / spatial_norm
        ranges = ranges / spatial_norm

        cube_shape = [*b_slices.shape[:-2], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        ranges = ranges[r]
        errors = errors[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path, }

        # add height ranges if provided
        if use_height_range:
            ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
            np.save(ranges_npy_path, ranges)
            batches_path['height_ranges'] = ranges_npy_path

        # add error ranges if provided
        if error_slices is not None:
            err_npy_path = os.path.join(work_directory, 'errors.npy')
            np.save(err_npy_path, errors)
            batches_path['errors'] = err_npy_path

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size, buffer=buffer)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size, strides=validation_strides)
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        return [boundary_loader, cube_loader]

class SynopticDataModule(LightningDataModule):

    def __init__(self, synoptic_files, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        # load synchronic map
        synoptic_r_map = Map(synoptic_files['Br'])
        synoptic_t_map = Map(synoptic_files['Bt'])
        synoptic_p_map = Map(synoptic_files['Bp'])

        synchronic_spherical_coords = all_coordinates_from_map(synoptic_r_map)
        synchronic_spherical_coords = np.stack([
            synchronic_spherical_coords.radius.value,
            np.pi / 2 + synchronic_spherical_coords.lat.to(u.rad).value,
            synchronic_spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        synchronic_coords = spherical_to_cartesian(synchronic_spherical_coords)

        synchronic_b_spherical = np.stack([synoptic_r_map.data, -synoptic_t_map.data, synoptic_p_map.data]).transpose()
        synchronic_b = vector_spherical_to_cartesian(synchronic_b_spherical, synchronic_spherical_coords)
        synchronic_transform = cartesian_to_spherical_matrix(synchronic_spherical_coords)

        b_spherical_slices = [synchronic_b_spherical]
        b_slices = [synchronic_b]
        error_slices = [np.zeros_like(synchronic_b)]
        coords = [synchronic_coords]
        spherical_coords = [synchronic_spherical_coords]
        transform = [synchronic_transform]

        if boundary['type'] == 'potential':
            source_surface_height = boundary['source_surface_height'] if 'source_surface_height' in boundary else 2.5
            resample = boundary['resample'] if 'resample' in boundary else [360, 180]
            sampling_points = boundary['sampling_points'] if 'sampling_points' in boundary else 100
            assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

            # PFSS extrapolation
            potential_r_map = Map(boundary['Br'])
            potential_r_map = potential_r_map.resample(resample * u.pix)
            potential_r_map.data[np.isnan(potential_r_map.data)] = 0
            pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
            pfss_out = pfsspy.pfss(pfss_in)

            # load B field
            ref_coords = all_coordinates_from_map(potential_r_map)
            spherical_boundary_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=height * u.solRad, frame=ref_coords.frame)
            potential_shape = spherical_boundary_coords.shape # required workaround for pfsspy spherical reshape
            spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
            spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
            spherical_boundary_values[..., 1] *= -1 # flip B_theta
            spherical_boundary_values = np.stack([spherical_boundary_values[..., 0],
                                                  spherical_boundary_values[..., 1],
                                                  spherical_boundary_values[..., 2]]).T

            # load coordinates
            spherical_boundary_coords = np.stack([
                spherical_boundary_coords.radius.value,
                np.pi / 2 + spherical_boundary_coords.lat.to(u.rad).value,
                spherical_boundary_coords.lon.to(u.rad).value]).T

            # convert to spherical coordinates
            boundary_values = vector_spherical_to_cartesian(spherical_boundary_values, spherical_boundary_coords)
            boundary_coords = spherical_to_cartesian(spherical_boundary_coords)
            boundary_transform = cartesian_to_spherical_matrix(spherical_boundary_coords)

            b_spherical_slices += [spherical_boundary_values]
            b_slices += [boundary_values]
            error_slices += [np.zeros_like(boundary_values)]
            coords += [boundary_coords]
            spherical_coords += [spherical_boundary_coords]
            transform += [boundary_transform]

        dataset_kwargs = {}

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')
            for b in b_spherical_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview Spherical": fig})
                plt.close('all')
            for c in coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_spherical_slices]).astype(np.float32)
        errors = np.concatenate([e.reshape((-1, 3)) for e in error_slices]).astype(np.float32)


        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            errors = errors[~nan_mask]

        # normalize data
        values = values / b_norm
        errors = errors / b_norm

        self.cube_shape = {'type': 'spherical', 'height': height}

        # check data
        assert len(coords) == len(transform) == len(values) == len(errors), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        errors = errors[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        err_npy_path = os.path.join(work_directory, 'errors.npy')
        np.save(err_npy_path, errors.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        'errors': err_npy_path
                        }

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings)
                                for settings in plot_settings if settings['type'] == 'slices'}
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        slices_loaders = [DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False) for ds in self.slices_datasets.values()]
        return boundary_loader, cube_loader, *slices_loaders

class PotentialTestDataModule(LightningDataModule):

    def __init__(self, synoptic_files, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)


        source_surface_height = boundary['source_surface_height'] if 'source_surface_height' in boundary else 2.5
        resample = boundary['resample'] if 'resample' in boundary else [360 * 2, 180 * 2]
        sampling_points = boundary['sampling_points'] if 'sampling_points' in boundary else 100
        assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

        # PFSS extrapolation
        potential_r_map = Map(synoptic_files['Br'])
        potential_r_map = potential_r_map.resample(resample * u.pix)
        potential_r_map.data[np.isnan(potential_r_map.data)] = 0
        pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
        pfss_out = pfsspy.pfss(pfss_in)

        # load B field
        ref_coords = all_coordinates_from_map(potential_r_map)
        spherical_boundary_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=1 * u.solRad, frame=ref_coords.frame)
        potential_shape = spherical_boundary_coords.shape # required workaround for pfsspy spherical reshape
        spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
        spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
        spherical_boundary_values[..., 1] *= -1 # flip B_theta
        spherical_boundary_values = np.stack([spherical_boundary_values[..., 0],
                                              spherical_boundary_values[..., 1],
                                              spherical_boundary_values[..., 2]]).T

        # load coordinates
        spherical_boundary_coords = np.stack([
            spherical_boundary_coords.radius.value,
            np.pi / 2 + spherical_boundary_coords.lat.to(u.rad).value,
            spherical_boundary_coords.lon.to(u.rad).value]).T

        # convert to spherical coordinates
        boundary_values = vector_spherical_to_cartesian(spherical_boundary_values, spherical_boundary_coords)
        boundary_coords = spherical_to_cartesian(spherical_boundary_coords)
        boundary_transform = cartesian_to_spherical_matrix(spherical_boundary_coords)

        b_spherical_slices = [spherical_boundary_values]
        b_slices = [boundary_values]
        error_slices = [np.zeros_like(boundary_values)]
        coords = [boundary_coords]
        spherical_coords = [spherical_boundary_coords]
        transform = [boundary_transform]

        dataset_kwargs = {}

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')
            for b in b_spherical_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview Spherical": fig})
                plt.close('all')
            for c in coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_spherical_slices]).astype(np.float32)
        errors = np.concatenate([e.reshape((-1, 3)) for e in error_slices]).astype(np.float32)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            errors = errors[~nan_mask]

        # normalize data
        values = values / b_norm
        errors = errors / b_norm

        self.cube_shape = {'type': 'spherical', 'height': height}

        # check data
        assert len(coords) == len(transform) == len(values) == len(errors), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        errors = errors[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        err_npy_path = os.path.join(work_directory, 'errors.npy')
        np.save(err_npy_path, errors.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        'errors': err_npy_path
                        }

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings)
                                for settings in plot_settings if settings['type'] == 'slices'}
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        slices_loaders = [DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False) for ds in self.slices_datasets.values()]
        return boundary_loader, cube_loader, *slices_loaders

class SphericalDataModule(LightningDataModule):

    def __init__(self, synoptic_files, full_disk_files, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        # load synchronic map
        synoptic_r_map = Map(synoptic_files['Br'])
        synoptic_t_map = Map(synoptic_files['Bt'])
        synoptic_p_map = Map(synoptic_files['Bp'])

        synchronic_spherical_coords = all_coordinates_from_map(synoptic_r_map)
        synchronic_spherical_coords = np.stack([
            synchronic_spherical_coords.radius.value,
            np.pi / 2 + synchronic_spherical_coords.lat.to(u.rad).value,
            synchronic_spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        synchronic_coords = spherical_to_cartesian(synchronic_spherical_coords)

        synchronic_b_spherical = np.stack([synoptic_r_map.data, -synoptic_t_map.data, synoptic_p_map.data]).transpose()
        synchronic_b = vector_spherical_to_cartesian(synchronic_b_spherical, synchronic_spherical_coords)
        synchronic_transform = cartesian_to_spherical_matrix(synchronic_spherical_coords)

        # load full disk map
        full_disk_r_map = Map(full_disk_files['Br'])
        full_disk_t_map = Map(full_disk_files['Bt'])
        full_disk_p_map = Map(full_disk_files['Bp'])

        full_disk_spherical_coords = all_coordinates_from_map(full_disk_r_map)
        full_disk_spherical_coords = full_disk_spherical_coords.transform_to(frames.HeliographicCarrington)
        full_disk_spherical_coords = np.stack([
            full_disk_spherical_coords.radius.to(u.solRad).value,
            np.pi / 2 + full_disk_spherical_coords.lat.to(u.rad).value,
            full_disk_spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        full_disk_coords = spherical_to_cartesian(full_disk_spherical_coords)
        full_disk_transform = cartesian_to_spherical_matrix(full_disk_spherical_coords)

        full_disk_b_spherical = np.stack(
            [full_disk_r_map.data, -full_disk_t_map.data, full_disk_p_map.data]).transpose()
        full_disk_b = vector_spherical_to_cartesian(full_disk_b_spherical, full_disk_spherical_coords)

        if 'Br_err' in full_disk_files and 'Bt_err' in full_disk_files and 'Bp_err' in full_disk_files:
            full_disk_r_error_map = Map(full_disk_files['Br_err'])
            full_disk_t_error_map = Map(full_disk_files['Bt_err'])
            full_disk_p_error_map = Map(full_disk_files['Bp_err'])
            full_disk_b_error = np.stack([full_disk_r_error_map.data,
                                          full_disk_t_error_map.data,
                                          full_disk_p_error_map.data]).transpose()
        else:
            full_disk_b_error = np.zeros_like(full_disk_b)

        # mask overlap
        synoptic_r_map.meta['date-obs'] = full_disk_r_map.meta['date-obs']  # set constant background
        reprojected_map = full_disk_r_map.reproject_to(synoptic_r_map.wcs)
        mask = ~np.isnan(reprojected_map.data).T
        synchronic_b_spherical[mask] = np.nan
        synchronic_b[mask] = np.nan

        b_spherical_slices = [synchronic_b_spherical, full_disk_b_spherical]
        b_slices = [synchronic_b, full_disk_b]
        error_slices = [np.zeros_like(synchronic_b), full_disk_b_error]
        coords = [synchronic_coords, full_disk_coords]
        spherical_coords = [synchronic_spherical_coords, full_disk_spherical_coords]
        transform = [synchronic_transform, full_disk_transform]

        if boundary['type'] == 'potential':
            source_surface_height = boundary['source_surface_height'] if 'source_surface_height' in boundary else 2.5
            resample = boundary['resample'] if 'resample' in boundary else [360, 180]
            sampling_points = boundary['sampling_points'] if 'sampling_points' in boundary else 100
            assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

            # PFSS extrapolation
            potential_r_map = Map(boundary['Br'])
            potential_r_map = potential_r_map.resample(resample * u.pix)
            pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
            pfss_out = pfsspy.pfss(pfss_in)

            # load B field
            ref_coords = all_coordinates_from_map(potential_r_map)
            spherical_boundary_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=height * u.solRad, frame=ref_coords.frame)
            potential_shape = spherical_boundary_coords.shape # required workaround for pfsspy spherical reshape
            spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
            spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
            spherical_boundary_values[..., 1] *= -1 # flip B_theta
            spherical_boundary_values = np.stack([spherical_boundary_values[..., 0],
                                                  spherical_boundary_values[..., 1],
                                                  spherical_boundary_values[..., 2]]).T

            # load coordinates
            spherical_boundary_coords = np.stack([
                spherical_boundary_coords.radius.value,
                np.pi / 2 + spherical_boundary_coords.lat.to(u.rad).value,
                spherical_boundary_coords.lon.to(u.rad).value]).T

            # convert to spherical coordinates
            boundary_values = vector_spherical_to_cartesian(spherical_boundary_values, spherical_boundary_coords)
            boundary_coords = spherical_to_cartesian(spherical_boundary_coords)
            boundary_transform = cartesian_to_spherical_matrix(spherical_boundary_coords)

            b_spherical_slices += [spherical_boundary_values]
            b_slices += [boundary_values]
            error_slices += [np.zeros_like(boundary_values)]
            coords += [boundary_coords]
            spherical_coords += [spherical_boundary_coords]
            transform += [boundary_transform]

        dataset_kwargs = {}
        if slice:
            if slice['frame'] == 'helioprojective':
                bottom_left = SkyCoord(slice['Tx'][0] * u.arcsec, slice['Ty'][0] * u.arcsec,
                                       frame=full_disk_r_map.coordinate_frame)
                top_right = SkyCoord(slice['Tx'][1] * u.arcsec, slice['Ty'][1] * u.arcsec,
                                     frame=full_disk_r_map.coordinate_frame)
                bottom_left = bottom_left.transform_to(frames.HeliographicCarrington)
                top_right = top_right.transform_to(frames.HeliographicCarrington)
                slice_lon = np.array([bottom_left.lon.to(u.rad).value, top_right.lon.to(u.rad).value])
                slice_lat = np.array([bottom_left.lat.to(u.rad).value, top_right.lat.to(u.rad).value]) + np.pi / 2
            elif slice['frame'] == 'heliographic_carrington':
                slice_lon = slice['longitude']
                slice_lat = slice['latitude']
            else:
                raise ValueError(f"Unknown slice type '{slice['type']}'")
            # set values outside lat lon range to nan
            for b, c in zip(b_spherical_slices, spherical_coords):
                mask =  (c[..., 1] > slice_lat[0]) &\
                        (c[..., 1] < slice_lat[1]) &\
                        (c[..., 2] > slice_lon[0]) &\
                        (c[..., 2] < slice_lon[1])
                b[~mask] = np.nan
                c[~mask] = np.nan
            dataset_kwargs['latitude_range'] = slice_lat
            dataset_kwargs['longitude_range'] = slice_lon
            self.sampling_range = [[1, height], slice_lat, slice_lon]

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')
            for b in b_spherical_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview Spherical": fig})
                plt.close('all')
            for c in coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # load dataset
        # assert len(height_mapping['z']) == b_slices.shape[2], 'Invalid height mapping configuration: z must have the same length as the number of slices'
        # for i, h in enumerate(height_mapping['z']):
        #     coords[:, :, i, 2] = h
        # ranges = np.zeros((*coords.shape[:-1], 2))
        # use_height_range = 'z_max' in height_mapping
        # if use_height_range:
        #     z1 = height_mapping['z_max']
        #     # set to lower boundary if not specified
        #     z0 = height_mapping['z_min'] if 'z_min' in height_mapping else np.zeros_like(z1)
        #     assert len(z0) == len(z1) == len(height_mapping['z']), \
        #         'Invalid height mapping configuration: z_min, z_max and z must have the same length'
        #     for i, (h_min, h_max) in enumerate(zip(z0, z1)):
        #         ranges[:, :, i, 0] = h_min
        #         ranges[:, :, i, 1] = h_max

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_spherical_slices]).astype(np.float32)
        errors = np.concatenate([e.reshape((-1, 3)) for e in error_slices]).astype(np.float32)
        # ranges = ranges.reshape((-1, 2)).astype(np.float32)


        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            errors = errors[~nan_mask]
            # ranges = ranges[~nan_mask]

        # normalize data
        values = values / b_norm
        errors = errors / b_norm

        self.cube_shape = {'type': 'spherical', 'height': height}

        # check data
        assert len(coords) == len(transform) == len(values) == len(errors), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        errors = errors[r]
        # ranges = ranges[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        err_npy_path = os.path.join(work_directory, 'errors.npy')
        np.save(err_npy_path, errors.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        'errors': err_npy_path
                        }

        # add height ranges if provided
        # if use_height_range:
        #     ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        #     np.save(ranges_npy_path, ranges)
        #     batches_path['height_ranges'] = ranges_npy_path


        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings)
                                for settings in plot_settings if settings['type'] == 'slices'}
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        slices_loaders = [DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False) for ds in self.slices_datasets.values()]
        return boundary_loader, cube_loader, *slices_loaders


class AzimuthDataModule(LightningDataModule):

    def __init__(self, B_data, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        # load synchronic map
        B_field_map = Map(B_data['B_field'])
        B_az_map = Map(B_data['B_azimuth'])
        B_disamb_map = Map(B_data['B_disambig']) if 'B_disambig' in B_data else None
        B_in_map = Map(B_data['B_inclination'])

        fld = B_field_map.data
        inc = np.deg2rad(B_in_map.data)
        azi = np.deg2rad(B_az_map.data)

        # disambiguate
        if B_disamb_map is not None:
            amb = B_disamb_map.data
            amb_weak = 2
            condition = (amb.astype((int)) >> amb_weak).astype(bool)
            azi[condition] += np.pi

        azi -= np.pi

        spherical_coords = all_coordinates_from_map(B_field_map).transform_to(frames.HeliographicCarrington)
        radius = spherical_coords.radius
        spherical_coords = np.stack([
            radius.value if radius.isscalar else radius.to(u.solRad).value,
            np.pi / 2 + spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        coords = spherical_to_cartesian(spherical_coords)

        pAng = -np.deg2rad(B_field_map.meta['CROTA2'])
        latc, lonc = np.deg2rad(B_field_map.meta['CRLT_OBS']), np.deg2rad(B_field_map.meta['CRLN_OBS'])
        map_coords = all_coordinates_from_map(B_field_map).transform_to(frames.HeliographicCarrington)
        lat, lon = map_coords.lat.to(u.rad).value.transpose(), map_coords.lon.to(u.rad).value.transpose()

        cs_matrix = cartesian_to_spherical_matrix(spherical_coords)
        is_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
        si_matrix = np.linalg.inv(is_matrix)
        transform = np.matmul(si_matrix, cs_matrix)

        b_los = fld * np.cos(inc)
        b_trv = np.abs(fld * np.sin(inc))
        b = np.stack([b_los, b_trv, azi]).transpose()

        # load observer cartesian coordinates
        obs_coords = np.array([
            B_field_map.observer_coordinate.radius.to(u.solRad).value,
            np.pi / 2 + latc,
            lonc,
        ])
        obs_coords = spherical_to_cartesian(obs_coords)

        # define height mapping range
        if 'height_mapping' in B_data:
            height_mapping = B_data['height_mapping']
            min_coords = coords - obs_coords[None, None, :]

            # solve quadratic equation --> find points at min solar radii
            rays_d = coords - obs_coords[None, None, :]
            rays_o = obs_coords[None, None, :]
            a = rays_d.pow(2).sum(-1)
            b = (2 * rays_o * rays_d).sum(-1)
            c = rays_o.pow(2).sum(-1) - height_mapping["min"] ** 2
            dist_far = (-b - np.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

            a = rays_d.pow(2).sum(-1)
            b = (2 * rays_o * rays_d).sum(-1)
            c = rays_o.pow(2).sum(-1) - height_mapping["max"] ** 2
            dist_near = (-b - np.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

            height_range = np.stack([dist_near, dist_far], axis=-1)

        b_slices = [b]
        coords_slices = [coords]
        spherical_coords_slices = [spherical_coords]
        transforms = [transform]
        # observer_slices = [np.ones_like(coords) * obs_coords[None, :]]
        # height_range_slices = [height_range]

        dataset_kwargs = {'latitude_range': [np.nanmin(spherical_coords[..., 1]), np.nanmax(spherical_coords[..., 1])],
                          'longitude_range': [np.nanmin(spherical_coords[..., 2]), np.nanmax(spherical_coords[..., 2])],}
        if slice:
            if slice['frame'] == 'helioprojective':
                bottom_left = SkyCoord(slice['Tx'][0] * u.arcsec, slice['Ty'][0] * u.arcsec,
                                       frame=B_field_map.coordinate_frame)
                top_right = SkyCoord(slice['Tx'][1] * u.arcsec, slice['Ty'][1] * u.arcsec,
                                     frame=B_field_map.coordinate_frame)
                bottom_left = bottom_left.transform_to(frames.HeliographicCarrington)
                top_right = top_right.transform_to(frames.HeliographicCarrington)
                slice_lon = np.array([bottom_left.lon.to(u.rad).value, top_right.lon.to(u.rad).value])
                slice_lat = np.array([bottom_left.lat.to(u.rad).value, top_right.lat.to(u.rad).value]) + np.pi / 2
            elif slice['frame'] == 'heliographic_carrington':
                slice_lon = slice['longitude']
                slice_lat = slice['latitude']
            else:
                raise ValueError(f"Unknown slice type '{slice['type']}'")
            # set values outside lat lon range to nan
            for b, c, sc in zip(b_slices, coords_slices, spherical_coords_slices):
                mask = (sc[..., 1] > slice_lat[0]) & \
                       (sc[..., 1] < slice_lat[1]) & \
                       (sc[..., 2] > slice_lon[0]) & \
                       (sc[..., 2] < slice_lon[1])
                b[~mask] = np.nan
                c[~mask] = np.nan
            dataset_kwargs['latitude_range'] = slice_lat
            dataset_kwargs['longitude_range'] = slice_lon
            self.sampling_range = [[1, height], slice_lat, slice_lon]

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[0].set_title('$B_{LOS}$')
                axs[0].set_xlabel('Longitude [rad]'), axs[0].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[G]')
                im = axs[1].imshow(b[..., 1].transpose(), vmin=0, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].set_title('$B_{trv}$')
                axs[1].set_xlabel('Longitude [rad]'), axs[1].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[G]')
                axs[2].imshow(b[..., 2].transpose(), vmin=-np.pi, vmax=np.pi, cmap='gray', origin='lower')
                axs[2].set_title('$\phi_B$')
                axs[2].set_xlabel('Longitude [rad]'), axs[2].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[rad]')
                fig.tight_layout()
                wandb.log({"Overview": fig})
                plt.close('all')
            for c in coords_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                fig.tight_layout()
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                fig.tight_layout()
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords_slices]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transforms]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_slices]).astype(np.float32)
        # observer = np.concatenate([o.reshape((-1, 3)) for o in observer_slices]).astype(np.float32)
        # height_range = np.concatenate([h.reshape((-1, 2)) for h in height_range_slices]).astype(np.float32)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]

        # normalize data
        values[..., 0] /= b_norm
        values[..., 1] /= b_norm

        self.cube_shape = {'type': 'spherical', 'height': height, 'shapes': [b.shape[:2] for b in b_slices]}

        # check data
        assert len(coords) == len(transform) == len(values), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        # observer = observer[r]
        # height_range = height_range[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        # observer_npy_path = os.path.join(work_directory, 'observer.npy')
        # np.save(observer_npy_path, observer.astype(np.float32))
        # height_range_npy_path = os.path.join(work_directory, 'height_range.npy')
        # np.save(height_range_npy_path, height_range.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        # 'observer': observer_npy_path,
                        # 'height_range': height_range_npy_path
                        }

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        # update plot settings with dataset kwargs
        plot_settings = [{**dataset_kwargs, **settings} for settings in plot_settings]
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings, batch_size=boundary_batch_size)
                                for settings in plot_settings if settings['type'] == 'slices'}
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        slices_loaders = [DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False) for ds in self.slices_datasets.values()]
        return boundary_loader, cube_loader, *slices_loaders

class SHARPDataModule(SlicesDataModule):

    def __init__(self, data_path, bin=1, slice=None, *args, **kwargs):
        if isinstance(data_path, str):
            hmi_p = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))[0]  # x
            hmi_t = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))[0]  # y
            hmi_r = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))[0]  # z
            err_p = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))[0]  # x
            err_t = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))[0]  # y
            err_r = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))[0]  # z
        else:
            hmi_p, err_p, hmi_t, err_t, hmi_r, err_r = data_path
        # laod maps
        p_map, t_map, r_map = Map(hmi_p), Map(hmi_t), Map(hmi_r)
        p_error_map, t_error_map, r_error_map = Map(err_p), Map(err_t), Map(err_r)

        maps = [p_map, t_map, r_map, p_error_map, t_error_map, r_error_map]
        if slice:
            maps = [m.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                             top_right=u.Quantity((slice[1], slice[3]), u.pixel)) for m in maps]
        if bin > 1:
            maps = [m.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean) for m in maps]

        hmi_data = np.stack([maps[0].data, -maps[1].data, maps[2].data]).transpose()
        error_data = np.stack([maps[3].data, maps[4].data, maps[5].data]).transpose()

        b_slices = hmi_data[:, :, None]
        error_slices = error_data[:, :, None]
        meta_data = maps[0].meta

        super().__init__(b_slices, *args, error_slices=error_slices, meta_data=meta_data, **kwargs)


class SOLISDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, *args, **kwargs):
        dict_data = np.load(data_path, allow_pickle=True)
        sharp_cube = dict_data.item().get('sharp')
        vsm_cube = dict_data.item().get('vsm')
        vsm_cube = np.stack([np.ones_like(vsm_cube) * np.nan, np.ones_like(vsm_cube) * np.nan, vsm_cube])
        b_slices = np.stack([sharp_cube, vsm_cube], 1).T
        if slices is not None:
            b_slices = b_slices[:, :, slices]

        super().__init__(b_slices, *args, **kwargs)


class FITSDataModule(SlicesDataModule):

    def __init__(self, data, *args, **kwargs):
        # check if single height layer
        if not isinstance(data, list):
            data = [data]
        # load all heights
        b_list, range_list, error_list, reference_pixel_list, scale_list = [], [], [], [], []
        height_mapping = []
        for d in data:
            # LOAD B and COORDS
            # use center coordinate of first map
            b, meta_data = self.load_B(d['B'])
            error = self.load_error(d['B'])
            # FLIP SIGN
            if 'flip_sign' in d:
                b[..., d['flip_sign']] *= -1
            # APPLY MASK
            if "mask" in d:
                mask = self.load_mask(d["mask"])
                b[mask] = np.nan
            # LOAD HEIGHT MAPPING
            z, z_min, z_max = self.get_height_mapping(d)
            # flatten data
            b_list += [b]
            error_list += [error]
            height_mapping += [{'z': z, 'z_min': z_min, 'z_max': z_max}]


        # stack data
        b_slices = np.stack(b_list, axis=2)
        error_slices = np.stack(error_list, axis=2) if None not in error_list else None
        height_mapping = {'z': [h['z'] for h in height_mapping],
                          'z_min': [h['z_min'] for h in height_mapping],
                          'z_max': [h['z_max'] for h in height_mapping]}
        super().__init__(b_slices, meta_data=meta_data, height_mapping=height_mapping, error_slices=error_slices, *args, **kwargs)

    def load_B(self, b_data):
        if isinstance(b_data, dict):
            x_map = Map(b_data['x']) if 'x' in b_data else None
            y_map = Map(b_data['y']) if 'y' in b_data else None
            z_map = Map(b_data['z']) if 'z' in b_data else None
        elif isinstance(b_data, str):
            data = fits.getdata(b_data)
            meta = fits.getheader(b_data)
            meta['naxis'] = 2
            x_map = Map(data[0], meta)
            y_map = Map(data[1], meta)
            z_map = Map(data[2], meta)
        else:
            raise NotImplementedError(f'Unknown data format for B: {type(b_data)}')
        # add missing components as NaNs
        maps = [x_map, y_map, z_map]
        ref_map = [m for m in maps if m is not None][0]
        nan_data = np.ones_like(ref_map.data) * np.nan

        x_data = x_map.data if x_map is not None else nan_data
        y_data = y_map.data if y_map is not None else nan_data
        z_data = z_map.data if z_map is not None else nan_data

        b = np.stack([x_data, y_data, z_data]).T
        meta_data = ref_map.meta
        return b, meta_data

    def load_error(self, b_data, target_scale=None):
        if isinstance(b_data, dict) and 'x_error' in b_data:
            x_map = Map(b_data['x_error'])
            y_map = Map(b_data['y_error'])
            z_map = Map(b_data['z_error'])
        elif 'error' in b_data:
            data = fits.getdata(b_data['error'])
            meta = fits.getheader(b_data['error'])
            x_map = Map(data, meta)
            y_map = Map(data, meta)
            z_map = Map(data, meta)
        else:
            return None
        b_error = np.stack([x_map.data, y_map.data, z_map.data]).T
        return b_error

    def get_height_mapping(self, d):
        if 'height_mapping' in d:
            z = d['height_mapping']['z']
            z_min = d['height_mapping']['z_min']
            z_max = d['height_mapping']['z_max']
        else:
            z, z_min, z_max = 0, 0, 0
        return z, z_min, z_max

    def load_mask(self, mask_data):
        mask = fits.getdata(mask_data).T
        return mask == 0


class SHARPSeriesDataModule(SHARPDataModule):

    def __init__(self, file_paths, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.file_paths = copy(file_paths)

        super().__init__(file_paths[0], *self.args, **self.kwargs)

    def train_dataloader(self):
        # re-initialize
        super().__init__(self.file_paths[0], *self.args, **self.kwargs)
        del self.file_paths[0]  # continue with next file in list
        return super().train_dataloader()

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


class NumpyDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, bin=1, use_bz=False, components=False, *args, **kwargs):
        b_slices = np.load(data_path)
        if slices:
            b_slices = b_slices[:, :, slices]
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        if use_bz:
            b_slices[:, :, 1:, 0] = None
            b_slices[:, :, 1:, 1] = None
        if components:
            for i, c in enumerate(components):
                filter = [i for i in [0, 1, 2] if i not in c]
                b_slices[:, :, i, filter] = None
        super().__init__(b_slices, *args, **kwargs)


class AnalyticDataModule(LightningDataModule):

    def __init__(self, case, height, spatial_norm, b_norm, work_directory, batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None, boundary={"type": "full"}, **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm
        self.meta_data = None

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        tau_surfaces = boundary['tau_surfaces'] if boundary['type'] == 'tau' else None
        if case == 1:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi / 4, tau_surfaces=tau_surfaces)
        elif case == 2:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15, tau_surfaces=tau_surfaces,
                                          resolution=[80, 80, 72])
        else:
            raise Exception(f'Invalid CASE {case}. Available cases are: [1, 2]')

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(6, 2))
            min_max = np.abs(b_cube[..., i, 0]).max()
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')

        if boundary['type'] == "full":
            coord_cube = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1)
            values = [b_cube[:, :, :1].reshape((-1, 3)), b_cube[:, :, -1:].reshape((-1, 3)),
                      b_cube[:, :1, :].reshape((-1, 3)), b_cube[:, -1:, :].reshape((-1, 3)),
                      b_cube[:1, :, :].reshape((-1, 3)), b_cube[-1:, :, :].reshape((-1, 3)), ]
            coords = [coord_cube[:, :, :1].reshape((-1, 3)), coord_cube[:, :, -1:].reshape((-1, 3)),
                      coord_cube[:, :1, :].reshape((-1, 3)), coord_cube[:, -1:, :].reshape((-1, 3)),
                      coord_cube[:1, :, :].reshape((-1, 3)), coord_cube[-1:, :, :].reshape((-1, 3)), ]
            #
            coords = np.concatenate(coords).astype(np.float32)
            values = np.concatenate(values).astype(np.float32)
        elif boundary['type'] == "potential":
            b_cube = b_cube[:, :, 0]
            coords, values, err = prep_b_data(b_cube, np.zeros_like(b_cube), height,
                                              potential_boundary=True,
                                              potential_strides=boundary['strides'])
        elif boundary['type'] == "open":
            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            height_slices = boundary['height_slices'] if 'height_slices' in boundary else 0
            b_cube = b_cube[:, :, height_slices]
            coords = coords[:, :, height_slices]
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
        elif boundary['type'] == "tau":

            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            ranges = np.zeros((*coords.shape[:3], 2), dtype=np.float32)

            height_mapping = {"z": [h / 2 for h in tau_surfaces],
                              "z_min": [0] * len(tau_surfaces),
                              "z_max": [h for h in tau_surfaces]}
            self.height_mapping = height_mapping
            for i, (z, z_min, z_max) in enumerate(
                    zip(height_mapping["z"], height_mapping["z_min"], height_mapping["z_max"])):
                ranges[:, :, i, 0] = z_min
                ranges[:, :, i, 1] = z_max
                coords[:, :, i, 2] = z
            #
            if boundary['use_LOS']:
                b_cube[:, :, 1:, 0] = None
                b_cube[:, :, 1:, 1] = None
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
            ranges = ranges.reshape((-1, 2)).astype(np.float32)
        else:
            raise Exception(f'Invalid boundary condition: {boundary}. Available options: ["full", "potential", "open"]')

        # normalize data
        values = values / b_norm
        # apply spatial normalization
        coords = coords / spatial_norm

        cube_shape = [*b_cube.shape[:2], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        # create data loaders
        batches_path = {'coords': coords_npy_path, 'values': values_npy_path}
        if boundary['type'] == "tau":  # add ranges
            ranges /= spatial_norm
            ranges = ranges[r]
            ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
            np.save(ranges_npy_path, ranges)
            batches_path['height_ranges'] = ranges_npy_path

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size)
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        return [cube_loader, boundary_loader]
