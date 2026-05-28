import os
import uuid

import numpy as np
import torch
from astropy import units as u
from torch.utils.data import Dataset

from nf2.data.util import spherical_to_cartesian


class NF2Dataset(Dataset):

    def __init__(self, requires_jacobian=True):
        super().__init__()
        self.requires_jacobian = bool(requires_jacobian)


class TensorsDataset(NF2Dataset):

    def __init__(self, tensors, batch_size, work_path, filter_nans=True, shuffle=True, ds_name=None,
                 requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        if len(tensors) == 0:
            raise ValueError('TensorsDataset requires at least one tensor.')
        n_samples = {v.shape[0] for v in tensors.values()}
        if len(n_samples) != 1:
            raise ValueError(f'All tensors must have the same first dimension, got {sorted(n_samples)}.')

        nan_mask = np.any([np.all(np.isnan(t), axis=tuple(range(1, t.ndim))) for t in tensors.values()], axis=0)
        if nan_mask.sum() > 0 and filter_nans:
            print(f'Filtering {nan_mask.sum()} nan entries')
            tensors = {k: v[~nan_mask] for k, v in tensors.items()}

        if shuffle:
            r = np.random.permutation(list(tensors.values())[0].shape[0])
            tensors = {k: v[r] for k, v in tensors.items()}

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        os.makedirs(work_path, exist_ok=True)
        self.file_paths = {}
        for k, v in tensors.items():
            file_path = os.path.join(work_path, f'{ds_name}_{k}.npy')
            np.save(file_path, v.astype(np.float32))
            self.file_paths[k] = file_path

        self.batch_size = int(batch_size)

    def __len__(self):
        ref_file = list(self.file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        return {k: np.copy(np.load(file_path, mmap_mode='r')[start:stop])
                for k, file_path in self.file_paths.items()}

    def clear(self):
        for file_path in self.file_paths.values():
            if os.path.exists(file_path):
                os.remove(file_path)


class CubeDataset(NF2Dataset):

    def __init__(self, coord_range, ds_per_pixel=1 / 128, batch_size=2 ** 13, requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        x_resolution = int((coord_range[0, 1] - coord_range[0, 0]) / ds_per_pixel)
        y_resolution = int((coord_range[1, 1] - coord_range[1, 0]) / ds_per_pixel)
        z_resolution = int((coord_range[2, 1] - coord_range[2, 0]) / ds_per_pixel)
        self.axes = [
            np.linspace(coord_range[0, 0], coord_range[0, 1], x_resolution, dtype=np.float32),
            np.linspace(coord_range[1, 0], coord_range[1, 1], y_resolution, dtype=np.float32),
            np.linspace(coord_range[2, 0], coord_range[2, 1], z_resolution, dtype=np.float32),
        ]
        self.coords_shape = tuple(len(axis) for axis in self.axes)
        self.batch_size = int(batch_size)
        self.n_coords = int(np.prod(self.coords_shape))

    def __len__(self):
        return int(np.ceil(self.n_coords / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min((idx + 1) * self.batch_size, self.n_coords)
        coord = _coordinate_batch(self.axes, start, stop)
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}


class RandomSphericalCoordinateDataset(NF2Dataset):

    def __init__(self, radius_range, batch_size, Mm_per_ds,
                 latitude_range=(-90, 90), longitude_range=(0, 360), unit='deg',
                 volume_uniform_sampling=True, length=None, requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        longitude_range = u.Quantity(longitude_range, unit).to_value(u.rad)
        latitude_range = u.Quantity(latitude_range, unit).to_value(u.rad)
        colatitude_range = sorted(np.pi / 2 - latitude_range)  # convert to colatitude

        self.radius_range = radius_range
        self.Mm_per_ds = Mm_per_ds
        self.colatitude_range = colatitude_range
        self.longitude_range = longitude_range
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor
        self.volume_uniform_sampling = volume_uniform_sampling
        self.length = int(length) if length is not None else 1

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        # r [1, height]
        h_r = self.radius_range
        if self.volume_uniform_sampling:
            r_min, r_max = np.min(h_r), np.max(h_r)
            random_coords[:, 0] = (r_min ** 3 + random_coords[:, 0] * (r_max ** 3 - r_min ** 3)) ** (1 / 3)
        else:
            random_coords[:, 0] = h_r[0] + random_coords[:, 0] * (h_r[1] - h_r[0])
        # theta [0, pi]
        if self.volume_uniform_sampling:
            lat_r = self.colatitude_range
            v_min, v_max = np.min(np.cos(lat_r)), np.max(np.cos(lat_r))
            random_coords[:, 1] = v_min + random_coords[:, 1] * (v_max - v_min)
            random_coords[:, 1] = torch.arccos(random_coords[:, 1])
        else:
            lat_r = self.colatitude_range
            random_coords[:, 1] = lat_r[0] + random_coords[:, 1] * (lat_r[1] - lat_r[0])
        # phi [0, 2pi]
        lon_r = self.longitude_range
        random_coords[:, 2] = lon_r[0] + random_coords[:, 2] * (lon_r[1] - lon_r[0])
        # convert to cartesian
        random_coords = spherical_to_cartesian(random_coords, f=torch)
        random_coords = random_coords * (1 * u.solRad).to_value(u.Mm) / self.Mm_per_ds
        return {'coords': random_coords}


class RandomRadialGroupedCoordinateDataset(NF2Dataset):

    def __init__(self, radius_range, batch_size, Mm_per_ds,
                 n_lat_lon_sample=128, radial_sampling_exponent=1,
                 latitude_range=(-90, 90), longitude_range=(0, 360), unit='deg', length=None,
                 requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        longitude_range = u.Quantity(longitude_range, unit).to_value(u.rad)
        latitude_range = u.Quantity(latitude_range, unit).to_value(u.rad)
        colatitude_range = sorted(np.pi / 2 - latitude_range)  # convert to colatitude

        self.length = int(length) if length is not None else 1

        self.radius_range = radius_range
        self.Mm_per_ds = Mm_per_ds
        self.colatitude_range = colatitude_range
        self.longitude_range = longitude_range
        self.batch_size = int(batch_size)
        self.float_tensor = torch.FloatTensor

        self.n_lat_lon_sample = int(n_lat_lon_sample)
        if self.n_lat_lon_sample <= 0:
            raise ValueError('n_lat_lon_sample must be a positive integer.')
        if self.batch_size % self.n_lat_lon_sample != 0:
            raise ValueError(
                f'batch_size ({self.batch_size}) must be divisible by '
                f'n_lat_lon_sample ({self.n_lat_lon_sample}).')
        self.radial_sample = self.batch_size // self.n_lat_lon_sample
        self.radial_sampling_exponent = float(radial_sampling_exponent)
        if self.radial_sampling_exponent <= 0:
            raise ValueError('radial_sampling_exponent must be positive.')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # radial coordinates
        h_r = self.radius_range
        r_coords = self.float_tensor(self.radial_sample, 1).uniform_()
        r_coords = r_coords ** self.radial_sampling_exponent
        r_min, r_max = np.min(h_r), np.max(h_r)
        r_coords = (r_min ** 3 + r_coords * (r_max ** 3 - r_min ** 3)) ** (1 / 3)

        # latitude coordinates
        lat_coords = self.float_tensor(self.n_lat_lon_sample, self.radial_sample, 1).uniform_()
        lat_r = self.colatitude_range
        v_min, v_max = np.min(np.cos(lat_r)), np.max(np.cos(lat_r))
        lat_coords = v_min + lat_coords * (v_max - v_min)
        lat_coords = torch.arccos(lat_coords)

        # longitude coordinates
        lon_coords = self.float_tensor(self.n_lat_lon_sample, self.radial_sample, 1).uniform_()
        lon_r = self.longitude_range
        lon_coords = lon_r[0] + lon_coords * (lon_r[1] - lon_r[0])

        # expand r coords
        r_coords = r_coords[None, :, :].repeat(self.n_lat_lon_sample, 1, 1)

        grouped_coords = torch.cat([r_coords, lat_coords, lon_coords], -1)

        # convert to cartesian
        grouped_cartesian_coords = spherical_to_cartesian(grouped_coords, f=torch)
        grouped_cartesian_coords = grouped_cartesian_coords * (1 * u.solRad).to_value(u.Mm) / self.Mm_per_ds

        coords = grouped_cartesian_coords.reshape(-1, 3)

        return {'coords': coords, 'grouped_coords': grouped_cartesian_coords}


class SphereDataset(NF2Dataset):

    def __init__(self, radius_range, Mm_per_ds, resolution=256, batch_size=1024,
                 latitude_range=(-90, 90), longitude_range=(0, 360), unit='deg',
                 requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        longitude_range = u.Quantity(longitude_range, unit).to_value(u.rad)
        latitude_range = u.Quantity(latitude_range, unit).to_value(u.rad)
        colatitude_range = sorted(np.pi / 2 - latitude_range)  # convert to spherical coordinates
        #
        ratio = (colatitude_range[1] - colatitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(resolution * ratio)
        self.axes = [
            np.linspace(longitude_range[0], longitude_range[1], resolution, dtype=np.float32),
            np.linspace(colatitude_range[0], colatitude_range[1], resolution_lat, dtype=np.float32),
            np.linspace(radius_range[0], radius_range[1], resolution, dtype=np.float32),
        ]
        self.component_axes = [2, 1, 0]
        self.coords_shape = tuple(len(axis) for axis in self.axes)
        self.batch_size = int(batch_size)
        self.n_coords = int(np.prod(self.coords_shape))
        self.coord_scale = (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

    def __len__(self):
        return int(np.ceil(self.n_coords / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min((idx + 1) * self.batch_size, self.n_coords)
        coord = _spherical_coordinate_batch(self.axes, self.component_axes, start, stop)
        coord = spherical_to_cartesian(coord) * self.coord_scale
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}


class SphereSlicesDataset(NF2Dataset):

    def __init__(self, radius_range, Mm_per_ds,
                 latitude_range=(-90, 90), longitude_range=(0, 360), unit='deg',
                 longitude_resolution=256, batch_size=1024, n_slices=5, requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        longitude_range = u.Quantity(longitude_range, unit).to_value(u.rad)
        latitude_range = u.Quantity(latitude_range, unit).to_value(u.rad)
        colatitude_range = sorted(np.pi / 2 - latitude_range)  # convert to spherical coordinates
        #
        ratio = (colatitude_range[1] - colatitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(longitude_resolution * ratio)
        radius_samples = np.exp(np.linspace(np.log(radius_range[0]), np.log(radius_range[1]), n_slices))
        self.axes = [
            radius_samples.astype(np.float32, copy=False),
            np.linspace(colatitude_range[0], colatitude_range[1], resolution_lat, dtype=np.float32),
            np.linspace(longitude_range[0], longitude_range[1], longitude_resolution, dtype=np.float32),
        ]
        self.component_axes = [0, 1, 2]
        self.cube_shape = tuple(len(axis) for axis in self.axes)
        self.batch_size = int(batch_size)
        self.n_coords = int(np.prod(self.cube_shape))
        self.coord_scale = (1 * u.solRad).to_value(u.Mm) / Mm_per_ds

    def __len__(self):
        return int(np.ceil(self.n_coords / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min((idx + 1) * self.batch_size, self.n_coords)
        coord = _spherical_coordinate_batch(self.axes, self.component_axes, start, stop)
        coord = spherical_to_cartesian(coord) * self.coord_scale
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}


class SlicesDataset(NF2Dataset):

    def __init__(self, coord_range, ds_per_pixel, n_slices=10, batch_size=4096, requires_jacobian=True, **kwargs):
        super().__init__(requires_jacobian=requires_jacobian)
        x_resolution = int((coord_range[0, 1] - coord_range[0, 0]) / ds_per_pixel)
        y_resolution = int((coord_range[1, 1] - coord_range[1, 0]) / ds_per_pixel)
        if coord_range[2, 0] == 0:
            z_range = np.linspace(coord_range[2, 0], coord_range[2, 1], n_slices, dtype=np.float32)
        else:
            z_range = np.linspace(0, coord_range[2, 1], n_slices - 1, dtype=np.float32)
            z_range = np.concatenate([np.array([coord_range[2, 0]]), z_range])
        self.axes = [
            np.linspace(coord_range[0, 0], coord_range[0, 1], x_resolution, dtype=np.float32),
            np.linspace(coord_range[1, 0], coord_range[1, 1], y_resolution, dtype=np.float32),
            z_range.astype(np.float32, copy=False),
        ]
        self.cube_shape = tuple(len(axis) for axis in self.axes)
        self.batch_size = int(batch_size)
        self.n_coords = int(np.prod(self.cube_shape))

        super().__init__()

    def __len__(self):
        return int(np.ceil(self.n_coords / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min((idx + 1) * self.batch_size, self.n_coords)
        coord = _coordinate_batch(self.axes, start, stop)
        coord = torch.tensor(coord, dtype=torch.float32)
        return {'coords': coord}


def _coordinate_batch(axes, start, stop):
    shape = tuple(len(axis) for axis in axes)
    flat_idx = np.arange(start, stop, dtype=np.int64)
    indices = np.unravel_index(flat_idx, shape)
    return np.stack([axis[index] for axis, index in zip(axes, indices)], -1)


def _spherical_coordinate_batch(axes, component_axes, start, stop):
    shape = tuple(len(axis) for axis in axes)
    flat_idx = np.arange(start, stop, dtype=np.int64)
    indices = np.unravel_index(flat_idx, shape)
    return np.stack([axes[axis_idx][indices[axis_idx]] for axis_idx in component_axes], -1)


class RandomCoordinateDataset(NF2Dataset):

    def __init__(self, coord_range, batch_size=2 ** 14, buffer=None, z_sampling_exponent=1, length=None,
                 requires_jacobian=True):
        super().__init__(requires_jacobian=requires_jacobian)
        if buffer:
            buffer_x = (coord_range[0, 1] - coord_range[0, 0]) * buffer
            buffer_y = (coord_range[1, 1] - coord_range[1, 0]) * buffer
            coord_range[0, 0] -= buffer_x
            coord_range[0, 1] += buffer_x
            coord_range[1, 0] -= buffer_y
            coord_range[1, 1] += buffer_y
        self.coord_range = coord_range
        self.batch_size = int(batch_size)
        self.float_tensor = torch.FloatTensor
        self.z_sampling_exponent = torch.tensor(z_sampling_exponent, dtype=torch.float32)
        self.length = length if length is not None else 1

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        random_coords[:, 0] = (
                random_coords[:, 0] * (self.coord_range[0, 1] - self.coord_range[0, 0]) + self.coord_range[0, 0])
        random_coords[:, 1] = (
                random_coords[:, 1] * (self.coord_range[1, 1] - self.coord_range[1, 0]) + self.coord_range[1, 0])
        random_coords[:, 2] = random_coords[:, 2] ** self.z_sampling_exponent
        random_coords[:, 2] = (
                random_coords[:, 2] * (self.coord_range[2, 1] - self.coord_range[2, 0]) + self.coord_range[2, 0])
        return {'coords': random_coords}


class RandomHeightCoordinateDataset(NF2Dataset):

    def __init__(self, coord_range, batch_size=2 ** 14, z_sample=128, z_sampling_exponent=2, length=None,
                 requires_jacobian=True):
        super().__init__(requires_jacobian=requires_jacobian)
        self.coord_range = coord_range
        self.batch_size = int(batch_size)
        self.float_tensor = torch.FloatTensor
        self.z_sampling_exponent = torch.tensor(z_sampling_exponent, dtype=torch.float32)
        self.length = int(length) if length is not None else 1
        self.z_sample = z_sample

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # sample z coords
        random_z_coords = self.float_tensor(self.z_sample, 1).uniform_() ** self.z_sampling_exponent
        # scale z
        random_z_coords = (random_z_coords * (self.coord_range[2, 1] - self.coord_range[2, 0]) + self.coord_range[2, 0])
        # sample xy coords per z
        random_xy_coords = self.float_tensor(self.batch_size // self.z_sample, self.z_sample, 2).uniform_()
        random_xy_coords[..., 0] = (
                random_xy_coords[..., 0] * (self.coord_range[0, 1] - self.coord_range[0, 0]) + self.coord_range[0, 0])
        random_xy_coords[..., 1] = (
                random_xy_coords[..., 1] * (self.coord_range[1, 1] - self.coord_range[1, 0]) + self.coord_range[1, 0])
        random_z_coords = random_z_coords[None, :, :].repeat(self.batch_size // self.z_sample, 1, 1)

        z_grouped_coords = torch.cat([random_xy_coords, random_z_coords], -1)
        random_coords = z_grouped_coords.reshape(-1, 3)

        return {'coords': random_coords, 'grouped_coords': z_grouped_coords}


class IndexedDataset(Dataset):

    def __init__(self, dataset, key='dataset_idx'):
        """Data set wrapper to add an index to each data sample.

        :param dataset: base dataset.
        :param key: key to use for the index.
        """
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data[self.key] = torch.tensor([idx], dtype=torch.long)
        return data

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError
