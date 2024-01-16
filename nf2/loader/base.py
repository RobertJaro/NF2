import os

import numpy as np
import wandb
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sunpy.map import Map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset
from nf2.data.loader import load_potential_field_data


class SlicesDataModule(LightningDataModule):

    def __init__(self, b_slices, height, spatial_norm, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 error_slices=None, height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_strides=1,
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
        assert len(height_mapping['z']) == b_slices.shape[
            2], 'Invalid height mapping configuration: z must have the same length as the number of slices'
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
            b_bottom = np.nan_to_num(b_bottom, nan=0)  # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'],
                                                                        progress=True)
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
            b_bottom = np.nan_to_num(b_bottom, nan=0)  # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'],
                                                                        only_top=True, pf_error=0.1, progress=True)
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
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size,
                                        strides=validation_strides)
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
        super().__init__(b_slices, meta_data=meta_data, height_mapping=height_mapping, error_slices=error_slices, *args,
                         **kwargs)

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
