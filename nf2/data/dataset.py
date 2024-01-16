import numpy as np
import torch
from torch.utils.data import Dataset

from nf2.data.util import spherical_to_cartesian


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(
            np.load(list(self.batches_file_paths.values())[0], mmap_mode='r').shape[0] / self.batch_size).astype(
            np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data


class ImageDataset(Dataset):

    def __init__(self, cube_shape, norm, z=0):
        coordinates = np.stack(np.mgrid[:cube_shape[0],
                               :cube_shape[1]], -1)
        self.coordinates = coordinates
        self.coordinates_flat = coordinates.reshape((-1, 2))
        self.norm = norm
        self.z = z / self.norm

    def __len__(self, ):
        return self.coordinates_flat.shape[0]

    def __getitem__(self, idx):
        coord = self.coordinates_flat[idx]
        scaled_coord = [coord[0] / self.norm,
                        coord[1] / self.norm,
                        self.z]
        return np.array(scaled_coord, dtype=np.float32)


class CubeDataset(Dataset):

    def __init__(self, cube_shape, spatial_norm, strides=1, batch_size=1024):
        coords = np.stack(np.mgrid[:cube_shape[0]:strides, :cube_shape[1]:strides, :cube_shape[2]:strides], -1)
        self.coords_shape = coords.shape[:-1]
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords = coords.view((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return coord


class RandomSphericalCoordinateDataset(Dataset):

    def __init__(self, height_range, batch_size,
                 latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi),
                 radial_weighted_sampling=True, latitude_weighted_sampling=True):
        self.height_range = height_range
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor
        self.radial_weighted_sampling = radial_weighted_sampling
        self.latitude_weighted_sampling = latitude_weighted_sampling

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        # r [1, height]
        h_r = self.height_range
        if self.radial_weighted_sampling:
            v_min, v_max = np.min(np.log(h_r)), np.max(np.log(h_r))
            random_coords[:, 0] = v_min + random_coords[:, 0] * (v_max - v_min)
            random_coords[:, 0] = torch.exp(random_coords[:, 0])
        else:
            random_coords[:, 0] = h_r[0] + random_coords[:, 0] * (h_r[1] - h_r[0])
        # theta [0, pi]
        if self.latitude_weighted_sampling:
            lat_r = self.latitude_range
            v_min, v_max = np.min(np.cos(lat_r)), np.max(np.cos(lat_r))
            random_coords[:, 1] = v_min + random_coords[:, 1] * (v_max - v_min)
            random_coords[:, 1] = torch.arccos(random_coords[:, 1])
        else:
            lat_r = self.latitude_range
            random_coords[:, 1] = lat_r[0] + random_coords[:, 1] * (lat_r[1] - lat_r[0])
        # phi [0, 2pi]
        lon_r = self.longitude_range
        random_coords[:, 2] = lon_r[0] + random_coords[:, 2] * (lon_r[1] - lon_r[0])
        # convert to cartesian
        random_coords = spherical_to_cartesian(random_coords, f=torch)
        return random_coords


class SphereDataset(Dataset):

    def __init__(self, height_range, resolution=256, batch_size=1024, latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi)):
        ratio = (latitude_range[1] - latitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(resolution * ratio)
        coords = np.stack(
            np.meshgrid(np.linspace(height_range[0], height_range[1], resolution),
                        np.linspace(latitude_range[0], latitude_range[1], resolution_lat),
                        np.linspace(longitude_range[0], longitude_range[1], resolution),
                        indexing='ij')).T
        self.coords_shape = coords.shape[:-1]

        coords = spherical_to_cartesian(coords)
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.reshape((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return coord

class SphereSlicesDataset(Dataset):

    def __init__(self, radius_range=(1.0, 2.0), latitude_range=(0, np.pi), longitude_range=(0, 2 * np.pi), longitude_resolution=256, batch_size=1024, n_slices=5, **kwargs):
        ratio = (latitude_range[1] - latitude_range[0]) / (longitude_range[1] - longitude_range[0])
        resolution_lat = int(longitude_resolution * ratio)
        coords = np.stack(
            np.meshgrid(np.linspace(radius_range[0], radius_range[1], n_slices),
                        np.linspace(latitude_range[0], latitude_range[1], resolution_lat),
                        np.linspace(longitude_range[0], longitude_range[1], longitude_resolution),
                        indexing='ij')).T
        self.cube_shape = coords.shape[:-1]

        coords = spherical_to_cartesian(coords)
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.reshape((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return coord

class RandomCoordinateDataset(Dataset):

    def __init__(self, cube_shape, spatial_norm, batch_size, buffer=None):
        super().__init__()
        cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        if buffer:
            buffer_x = (cube_shape[0, 1] - cube_shape[0, 0]) * buffer
            buffer_y = (cube_shape[1, 1] - cube_shape[1, 0]) * buffer
            cube_shape[0, 0] -= buffer_x
            cube_shape[0, 1] += buffer_x
            cube_shape[1, 0] -= buffer_y
            cube_shape[1, 1] += buffer_y
        self.cube_shape = cube_shape
        self.spatial_norm = spatial_norm
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        random_coords[:, 0] = (
                    random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (
                    random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (
                    random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords /= self.spatial_norm
        return random_coords
