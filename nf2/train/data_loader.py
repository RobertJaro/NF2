import glob
import os
from copy import copy

import numpy as np
import pfsspy
import wandb
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_heliographic_header
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset, RandomSphericalCoordinateDataset, \
    SphereDataset
from nf2.data.loader import prep_b_data, load_potential_field_data
from astropy import units as u

from nf2.data.util import vector_spherical_to_cartesian, spherical_to_cartesian, vector_cartesian_to_spherical, \
    cartesian_to_spherical_matrix


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
        return [cube_loader, boundary_loader]



class SphericalDataModule(LightningDataModule):

    def __init__(self, synoptic_files, full_disk_files, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 error_slices=None, height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_resolution = 256,
                 meta_data=None, plot_overview=True, Mm_per_pixel=None, buffer=None,
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        self.Mm_per_pixel = Mm_per_pixel

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
            np.deg2rad(synchronic_spherical_coords.lon.value),
            np.pi / 2 + np.deg2rad(synchronic_spherical_coords.lat.value),
            synchronic_spherical_coords.radius.value]).transpose()
        synchronic_coords = spherical_to_cartesian(synchronic_spherical_coords)

        synchronic_b_spherical = np.stack([synoptic_p_map.data, -synoptic_t_map.data, synoptic_r_map.data]).transpose()
        synchronic_b = vector_spherical_to_cartesian(synchronic_b_spherical, synchronic_spherical_coords)
        synchronic_transform = cartesian_to_spherical_matrix(synchronic_coords)

        # load full disk map
        full_disk_r_map = Map(full_disk_files['Br'])
        full_disk_t_map = Map(full_disk_files['Bt'])
        full_disk_p_map = Map(full_disk_files['Bp'])

        full_disk_spherical_coords = all_coordinates_from_map(full_disk_r_map)
        full_disk_spherical_coords = full_disk_spherical_coords.transform_to(frames.HeliographicCarrington)
        full_disk_spherical_coords = np.stack([
            np.deg2rad(full_disk_spherical_coords.lon.value),
            np.pi / 2 + np.deg2rad(full_disk_spherical_coords.lat.value),
            full_disk_spherical_coords.radius.to(u.solRad).value]).transpose()
        full_disk_coords = spherical_to_cartesian(full_disk_spherical_coords)
        full_disk_transform = cartesian_to_spherical_matrix(full_disk_coords)


        full_disk_b_spherical = np.stack([full_disk_p_map.data, -full_disk_t_map.data, full_disk_r_map.data]).transpose()
        full_disk_b = vector_spherical_to_cartesian(full_disk_b_spherical, full_disk_spherical_coords)

        # mask overlap
        synoptic_r_map.meta['date-obs'] = full_disk_r_map.meta['date-obs'] # set constant background
        reprojected_map = full_disk_r_map.reproject_to(synoptic_r_map.wcs)
        mask = ~np.isnan(reprojected_map.data).T
        synchronic_b_spherical[mask] = np.nan
        synchronic_b[mask] = np.nan

        b_spherical_slices = [synchronic_b_spherical, full_disk_b_spherical]
        b_slices = [synchronic_b, full_disk_b]
        coords = [synchronic_coords, full_disk_coords]
        spherical_coords = [synchronic_spherical_coords, full_disk_spherical_coords]
        transform = [synchronic_transform, full_disk_transform]

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
        # ranges = ranges.reshape((-1, 2)).astype(np.float32)
        # errors = error_slices.reshape((-1, 3)).astype(np.float32) if error_slices is not None else np.zeros_like(values)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            # ranges = ranges[~nan_mask]
            # errors = errors[~nan_mask]

        if boundary['type'] == 'potential':
            potential_map = synoptic_r_map.resample(boundary['resample'] * u.pix)
            pfss_in = pfsspy.Input(potential_map, 50, height)
            pfss_out = pfsspy.pfss(pfss_in)
            boundary_values = pfss_out.bg[:, :, -1].value

            boundary_coords = all_coordinates_from_map(pfss_out.source_surface_br)
            boundary_coords = np.stack([
                np.deg2rad(boundary_coords.lon.value),
                np.pi / 2 + np.deg2rad(boundary_coords.lat.value),
                boundary_coords.radius.value]).transpose()
            boundary_transform = cartesian_to_spherical_matrix(boundary_coords)

            # log upper boundary
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            potential_norm = np.max(np.abs(boundary_values))
            axs[0].imshow(boundary_values[..., 0].transpose(), vmin=-potential_norm, vmax=potential_norm, cmap='gray', origin='lower')
            axs[1].imshow(boundary_values[..., 1].transpose(), vmin=-potential_norm, vmax=potential_norm, cmap='gray', origin='lower')
            axs[2].imshow(boundary_values[..., 2].transpose(), vmin=-potential_norm, vmax=potential_norm, cmap='gray', origin='lower')
            wandb.log({"Potential boundary": fig})
            plt.close('all')
            #
            values = np.concatenate([boundary_values.reshape((-1, 3)), values])
            coords = np.concatenate([boundary_coords.reshape((-1, 3)), coords])
            transform = np.concatenate([boundary_transform.reshape((-1, 3, 3)), transform])

        elif boundary['type'] == 'stress_free':
            boundary_values = np.zeros_like(synchronic_b, dtype=np.float32)
            boundary_coords = np.copy(synchronic_spherical_coords)
            boundary_values[..., 2] = np.nan # arbitrary z value
            boundary_coords[..., 2] = height # top boundary
            boundary_transform = cartesian_to_spherical_matrix(boundary_coords)
            boundary_coords = spherical_to_cartesian(boundary_coords)
            # concatenate boundary data points
            strides = boundary['strides'] if 'strides' in boundary else 1
            values = np.concatenate([boundary_values[::strides, ::strides].reshape((-1, 3)), values])
            coords = np.concatenate([boundary_coords[::strides, ::strides].reshape((-1, 3)), coords])
            transform = np.concatenate([boundary_transform[::strides, ::strides].reshape((-1, 3, 3)), transform])
        elif boundary['type'] == 'open':
            pass
        else:
            raise ValueError('Unknown boundary type')

        # normalize data
        values = values / b_norm
        # errors = errors / b_norm
        # apply spatial normalization
        # coords = coords / spatial_norm
        # ranges = ranges / spatial_norm

        self.cube_shape = {'type': 'spherical', 'height': height}

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        # ranges = ranges[r]
        # errors = errors[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path}

        # add height ranges if provided
        # if use_height_range:
        #     ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        #     np.save(ranges_npy_path, ranges)
        #     batches_path['height_ranges'] = ranges_npy_path

        # add error ranges if provided
        # if error_slices is not None:
        #     err_npy_path = os.path.join(work_directory, 'errors.npy')
        #     np.save(err_npy_path, errors)
        #     batches_path['errors'] = err_npy_path

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size)
        self.cube_dataset = SphereDataset(height, batch_size=boundary_batch_size, resolution=validation_resolution)
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        return [cube_loader, boundary_loader]



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
        p_map, t_map, r_map = Map(hmi_p),Map(hmi_t),Map(hmi_r)
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

    def __init__(self, data_paths, mask_path=None, bin=1, flip_sign=None, *args, **kwargs):
        # check if single height layer
        if isinstance(data_paths, dict):
            data_paths = [data_paths]
        # load all heights
        b_slices = []
        meta_data = None
        for data_path in data_paths:
            if isinstance(data_path, dict):
                data_dict = data_path
                b = np.stack([fits.getdata(data_dict['x']).T,
                              fits.getdata(data_dict['y']).T,
                              fits.getdata(data_dict['z']).T], -1)
                meta_data = fits.getheader(data_dict['x'])
            else:
                b = fits.getdata(data_path).T
                meta_data = fits.getheader(data_path)
            b_slices += [b]
        b_slices = np.stack(b_slices, 2)
        if flip_sign is not None:
            b_slices[..., flip_sign] *= -1  # -t component

        if mask_path is not None:
            mask = fits.getdata(mask_path).T
            b_slices[mask == 0, :, :] = np.nan
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        super().__init__(b_slices, meta_data=meta_data, *args, **kwargs)


class SHARPSeriesDataModule(SHARPDataModule):

    def __init__(self, file_paths, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.file_paths = copy(file_paths)

        super().__init__(file_paths[0], *self.args, **self.kwargs)

    def train_dataloader(self):
        # re-initialize
        super().__init__(self.file_paths[0], *self.args, **self.kwargs)
        del self.file_paths[0] # continue with next file in list
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

            height_mapping = {"z":  [h / 2 for h in tau_surfaces],
                                   "z_min": [0] * len(tau_surfaces),
                                   "z_max": [h for h in tau_surfaces]}
            self.height_mapping = height_mapping
            for i, (z, z_min, z_max) in enumerate(zip(height_mapping["z"], height_mapping["z_min"], height_mapping["z_max"])):
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




