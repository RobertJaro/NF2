import os
import uuid

import numpy as np
from astropy.nddata import block_reduce
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BatchesDataset
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

    def __init__(self, training_datasets, validation_datasets, module_config, num_workers=None, iterations=None):
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
        self.ds = np.array([
            np.diff(coords[:, 0, 0]).mean(),
            np.diff(coords[0, :, 1]).mean(),
        ])

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
