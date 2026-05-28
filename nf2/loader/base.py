import numpy as np
from astropy.nddata import block_reduce
from astropy.wcs import WCS
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from nf2.data.dataset import IndexedDataset, TensorsDataset
from nf2.loader.util import _plot_B, _plot_B_error, _plot_los_trv_azi


DEFAULT_NUM_WORKERS = 4


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

    def __init__(self, training_datasets, validation_datasets, module_config, num_workers=None,
                 train_num_workers=None, validation_num_workers=None,
                 persistent_workers=True, prefetch_factor=5):
        super().__init__()
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.config = module_config
        self.config.setdefault('schema_version', '0.4')
        self.config['training_dataset_ids'] = list(self.training_datasets.keys())
        self.config['validation_dataset_ids'] = list(self.validation_datasets.keys())
        self.config['datasets'] = {
            'training': {
                name: self._dataset_metadata(name, dataset)
                for name, dataset in self.training_datasets.items()
            },
            'validation': {
                name: self._dataset_metadata(name, dataset)
                for name, dataset in self.validation_datasets.items()
            },
        }
        self.validation_dataset_mapping = {i: name for i, name in enumerate(self.validation_datasets.keys())}
        self.num_workers = num_workers if num_workers is not None else DEFAULT_NUM_WORKERS
        self.train_num_workers = self.num_workers if train_num_workers is None else train_num_workers
        self.validation_num_workers = self.num_workers if validation_num_workers is None else validation_num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

    def _loader_kwargs(self, num_workers, shuffle=False, persistent_workers=False, prefetch_factor=None):
        kwargs = {
            'batch_size': None,
            'num_workers': int(num_workers),
            'pin_memory': False,
            'shuffle': shuffle,
        }
        if kwargs['num_workers'] > 0:
            kwargs['persistent_workers'] = bool(persistent_workers)
            if prefetch_factor is not None:
                kwargs['prefetch_factor'] = int(prefetch_factor)
        return kwargs

    @classmethod
    def _dataset_metadata(cls, name, dataset):
        metadata = {
            'id': name,
            'class': dataset.__class__.__name__,
            'requires_jacobian': bool(getattr(dataset, 'requires_jacobian', False)),
        }
        fields = [
            'config',
            'batch_size',
            'cube_shape',
            'coords_shape',
            'coord_range',
            'radius_range',
            'cartesian_radius_range',
            'coord_scale',
            'colatitude_range',
            'longitude_range',
            'ds_per_pixel',
            'ds',
            'height_mapping',
            'los_trv_azi',
            'Mm_per_ds',
            'z_sampling_exponent',
            'radial_sampling_exponent',
            'n_lat_lon_sample',
            'radial_sample',
            'length',
        ]
        for field in fields:
            if hasattr(dataset, field):
                metadata[field] = cls._metadata_value(getattr(dataset, field))
        if hasattr(dataset, 'wcs') and dataset.wcs is not None:
            metadata['wcs_header'] = cls._metadata_value(dataset.wcs)
        return metadata

    @classmethod
    def _metadata_value(cls, value):
        if isinstance(value, WCS):
            return value.to_header_string()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, 'detach') and hasattr(value, 'cpu'):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.numpy().tolist()
        if isinstance(value, dict):
            return {str(k): cls._metadata_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._metadata_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def clear(self):
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, TensorsDataset)]

    def train_dataloader(self):
        datasets = {
            name: RequiresJacobianDataset(ds)
            for name, ds in self.training_datasets.items()
        }
        loaders = {name: DataLoader(ds, **self._loader_kwargs(
                                    self.train_num_workers, shuffle=True,
                                    persistent_workers=self.persistent_workers,
                                    prefetch_factor=self.prefetch_factor))
                   for name, ds in datasets.items()}
        return CombinedLoader(loaders, 'max_size_cycle')

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            # add dataset wrapper for indexing
            dataset = IndexedDataset(dataset)
            loader = DataLoader(dataset, **self._loader_kwargs(self.validation_num_workers, shuffle=False))
            loaders.append(loader)
        return loaders


class MapDataset(TensorsDataset):

    def __init__(self, b, b_err=None, coords=None,
                 Gauss_per_dB=1000, Mm_per_pixel=0.36, Mm_per_ds=100,
                 bin=1, height_mapping=None, log_tau=None,
                 coordinate_center=None, center=None, origin=None,
                 plot=True, los_trv_azi=False, ambiguous_azimuth=False,
                 wcs=None, **kwargs):
        self.ds_per_pixel = (Mm_per_pixel * bin) / Mm_per_ds
        self.Mm_per_pixel = Mm_per_pixel * bin
        self.coordinate_center, center_axes = _coordinate_center_ds(coordinate_center, center, origin, Mm_per_ds)

        # binning
        b = block_reduce(b, (bin, bin, 1), np.mean)
        self.bz = np.copy(b[..., 0]) if los_trv_azi else np.copy(b[..., 2])

        # normalize
        if los_trv_azi:
            b[..., :2] /= Gauss_per_dB
        else:
            b /= Gauss_per_dB

        if coords is None:
            coords = np.stack(np.mgrid[:b.shape[0], :b.shape[1], :1], -1).astype(np.float32) * self.ds_per_pixel
            coords = coords[:, :, 0, :]  # flatten z dimension
            coords = _center_coords(coords, self.coordinate_center, center_axes)
        else:
            coords = coords / Mm_per_ds
            coords = _center_coords(coords, self.coordinate_center, center_axes)

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
                b_err[..., :2] /= Gauss_per_dB
            else:
                b_err /= Gauss_per_dB
            tensors['b_err'] = b_err

            if plot and not los_trv_azi:
                _plot_B_error(b * Gauss_per_dB, b_err * Gauss_per_dB, coords * Mm_per_ds)
        else:
            if plot and los_trv_azi:
                b_plot = np.copy(b)
                b_plot[..., :2] *= Gauss_per_dB
                _plot_los_trv_azi(b_plot, coords * Mm_per_ds)
            if plot and not los_trv_azi:
                _plot_B(b * Gauss_per_dB, coords * Mm_per_ds)

        # prepare azimuth data after plotting
        if los_trv_azi and ambiguous_azimuth:
            b[..., 2] = np.mod(b[..., 2], np.pi)

        tensors = {k: v.reshape((-1, *v.shape[2:])).astype(np.float32) for k, v in tensors.items()}

        super().__init__(tensors, **kwargs)


def _coordinate_center_ds(coordinate_center=None, center=None, origin=None, Mm_per_ds=100):
    values = [v for v in [coordinate_center, center, origin] if v is not None]
    if len(values) > 1:
        raise ValueError("Use only one of coordinate_center, center, or origin.")
    if not values:
        return np.zeros(3, dtype=np.float32), 2
    value = values[0]
    if isinstance(value, dict):
        value = [value.get("x", 0), value.get("y", 0), value.get("z", 0)]
    value = np.asarray(value, dtype=np.float32)
    if value.shape not in {(2,), (3,)}:
        raise ValueError("coordinate_center must be [x, y] or [x, y, z] in Mm.")
    center_axes = value.shape[0]
    if value.shape == (2,):
        value = np.array([value[0], value[1], 0], dtype=np.float32)
    return value / Mm_per_ds, center_axes


def _center_coords(coords, coordinate_center, center_axes=2):
    coords = np.array(coords, dtype=np.float32, copy=True)
    center_axes = min(coords.shape[-1], center_axes)
    for axis in range(center_axes):
        if axis == 2 and np.allclose(coords[..., axis], coords[..., axis].flat[0]):
            data_center = coords[..., axis].flat[0]
        else:
            data_center = (np.nanmin(coords[..., axis]) + np.nanmax(coords[..., axis])) / 2
        coords[..., axis] = coords[..., axis] - data_center + coordinate_center[axis]
    return coords
