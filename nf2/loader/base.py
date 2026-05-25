import os

import numpy as np
from astropy.nddata import block_reduce
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from nf2.data.dataset import IndexedDataset, TensorsDataset
from nf2.loader.util import _plot_B, _plot_B_error, _plot_los_trv_azi


class RequiresJacobianDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        batch['requires_jacobian'] = self.dataset.requires_jacobian
        return batch

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class BaseDataModule(LightningDataModule):

    def __init__(self, training_datasets, validation_datasets, module_config, num_workers=None):
        super().__init__()
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.config = module_config
        self.validation_dataset_mapping = {i: name for i, name in enumerate(self.validation_datasets.keys())}
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def clear(self):
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, TensorsDataset)]

    def train_dataloader(self):
        datasets = {
            name: RequiresJacobianDataset(ds)
            for name, ds in self.training_datasets.items()
        }
        loaders = {name: DataLoader(ds, batch_size=None, num_workers=self.num_workers,
                                    pin_memory=False, shuffle=True, persistent_workers=True, prefetch_factor=5)
                   for name, ds in datasets.items()}
        return CombinedLoader(loaders, 'max_size_cycle')

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            # add dataset wrapper for indexing
            dataset = IndexedDataset(dataset)
            loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=False,
                                shuffle=False)
            loaders.append(loader)
        return loaders


class MapDataset(TensorsDataset):

    def __init__(self, b, b_err=None, coords=None,
                 G_per_dB=2500, Mm_per_pixel=0.36, Mm_per_ds=.36 * 320,
                 bin=1, height_mapping=None, log_tau=None,
                 plot=True, los_trv_azi=False, ambiguous_azimuth=False,
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
            coords = coords[:, :, 0, :]  # flatten z dimension
            # shift coordinate system to center
            x_max, y_max = coords[..., 0].max(), coords[..., 1].max()
            coords[..., 0] -= x_max / 2
            coords[..., 1] -= y_max / 2
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
            coords[..., 2] = z / Mm_per_ds

            if 'z_min' in height_mapping and 'z_max' in height_mapping:
                z_min = height_mapping['z_min']
                z_max = height_mapping['z_max']
                ranges = np.zeros((*self.cube_shape, 2), dtype=np.float32)
                ranges[..., 0] = z_min / Mm_per_ds
                ranges[..., 1] = z_max / Mm_per_ds
                tensors['height_range'] = ranges

        if log_tau is not None:
            coords[..., 2] = log_tau

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
