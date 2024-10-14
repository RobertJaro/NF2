import os
import uuid

import numpy as np
import wandb
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset
from nf2.data.loader import load_potential_field_data
from nf2.loader.util import _plot_B, _plot_B_error, _plot_los_trv_azi


class TensorsDataset(BatchesDataset):

    def __init__(self, tensors, work_directory, filter_nans=True, shuffle=True, ds_name=None, **kwargs):
        # filter nan entries
        nan_mask = np.any([np.all(np.isnan(t), axis=tuple(range(1, t.ndim))) for t in tensors.values()], axis=0)
        if nan_mask.sum() > 0 and filter_nans:
            print(f'Filtering {nan_mask.sum()} nan entries')
            tensors = {k: v[~nan_mask] for k, v in tensors.items()}

        # shuffle data
        if shuffle:
            r = np.random.permutation(list(tensors.values())[0].shape[0])
            tensors = {k: v[r] for k, v in tensors.items()}

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        batches_paths = {}
        for k, v in tensors.items():
            coords_npy_path = os.path.join(work_directory, f'{ds_name}_{k}.npy')
            np.save(coords_npy_path, v.astype(np.float32))
            batches_paths[k] = coords_npy_path

        super().__init__(batches_paths, **kwargs)


class BaseDataModule(LightningDataModule):

    def __init__(self, training_datasets, validation_datasets, module_config, num_workers=None, iterations=None,
                 **kwargs):
        super().__init__()
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.config = module_config
        self.validation_dataset_mapping = {i: name for i, name in enumerate(self.validation_datasets.keys())}
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.iterations = iterations

    def clear(self):
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, BatchesDataset)]

    def train_dataloader(self):
        datasets = self.training_datasets

        # data loader with fixed number of iterations
        if self.iterations is not None:
            loaders = {}
            for i, (name, dataset) in enumerate(datasets.items()):
                sampler = RandomSampler(dataset, replacement=True, num_samples=int(self.iterations))
                loaders[name] = DataLoader(dataset, batch_size=None, num_workers=self.num_workers,
                                           pin_memory=True, sampler=sampler)
            return loaders

        # data loader with iterations based on the largest dataset
        ref_idx = np.argmax([len(ds) for ds in datasets.values()])
        ref_dataset_name, ref_dataset = list(datasets.items())[ref_idx]
        loaders = {ref_dataset_name: DataLoader(ref_dataset, batch_size=None, num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)}
        for i, (name, dataset) in enumerate(datasets.items()):
            if i == ref_idx:
                continue  # reference dataset already added
            sampler = RandomSampler(dataset, replacement=True, num_samples=len(ref_dataset))
            loaders[name] = DataLoader(dataset, batch_size=None, num_workers=self.num_workers,
                                       pin_memory=True, sampler=sampler)
        return loaders

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                shuffle=False)
            loaders.append(loader)
        return loaders


class MapDataset(TensorsDataset):

    def __init__(self, b, b_err=None, coords=None,
                 G_per_dB=2500, Mm_per_pixel=0.36, Mm_per_ds=.36 * 320,
                 bin=1, height_mapping=None, plot=True, los_trv_azi=False, ambiguous_azimuth=False,
                 wcs=None, **kwargs):
        self.ds_per_pixel = (Mm_per_pixel * bin) / Mm_per_ds

        # binning
        b = block_reduce(b, (bin, bin, 1), np.mean)
        self.bz = np.copy(b[..., 0]) if los_trv_azi else np.copy(b[..., 2])

        # normalize
        if los_trv_azi:
            b[..., :2] /= G_per_dB
        else:
            b /= G_per_dB

        if coords is None:
            coords = np.stack(np.mgrid[:b.shape[0], :b.shape[1], :1], -1).astype(np.float32) * self.ds_per_pixel
            coords = coords[:, :, 0, :]
        else:
            coords = coords / Mm_per_ds

        self.coord_range = np.array([[coords[..., 0].min(), coords[..., 0].max()],
                                     [coords[..., 1].min(), coords[..., 1].max()]])

        self.cube_shape = coords.shape[:-1]
        self.los_trv_azi = los_trv_azi
        self.height_mapping = height_mapping
        self.wcs = wcs

        tensors = {'b_true': b}

        if height_mapping is not None:
            z = height_mapping['z']
            z_min = height_mapping['z_min'] if 'z_min' in height_mapping else 0
            z_max = height_mapping['z_max'] if 'z_max' in height_mapping else 0

            coords[..., 2] = z / Mm_per_ds
            ranges = np.zeros((*self.cube_shape, 2), dtype=np.float32)
            ranges[..., 0] = z_min / Mm_per_ds
            ranges[..., 1] = z_max / Mm_per_ds
            tensors['height_range'] = ranges

        tensors['coords'] = coords

        if b_err is not None:
            b_err = block_reduce(b_err, (bin, bin, 1), np.mean)
            if los_trv_azi:
                b_err[..., :2] /= G_per_dB
            else:
                b_err /= G_per_dB
            tensors['b_err'] = b_err

            if plot and not los_trv_azi:
                _plot_B_error(b * G_per_dB, b_err * G_per_dB, coords * Mm_per_ds)
        else:
            if plot and los_trv_azi:
                b_plot = np.copy(b)
                b_plot[..., :2] *= G_per_dB
                _plot_los_trv_azi(b_plot, coords * Mm_per_ds)
            if plot and not los_trv_azi:
                _plot_B(b * G_per_dB, coords * Mm_per_ds)

        # prepare azimuth data after plotting
        if los_trv_azi and ambiguous_azimuth:
            b[..., 2] = np.mod(b[..., 2], np.pi)

        tensors = {k: v.reshape((-1, *v.shape[2:])).astype(np.float32) for k, v in tensors.items()}

        super().__init__(tensors, **kwargs)


class SlicesDataModule(BaseDataModule):

    def __init__(self, b_slices, work_directory, height=160, spatial_norm=160, b_norm=2500,
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
