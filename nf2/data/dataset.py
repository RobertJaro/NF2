import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(np.load(list(self.batches_file_paths.values())[0], mmap_mode='r').shape[0] / self.batch_size).astype(np.int32)

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
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords = coords.view((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return coord


class RandomCoordinateDataset(Dataset):

    def __init__(self, cube_shape, spatial_norm, batch_size):
        super().__init__()
        self.cube_shape = cube_shape
        self.spatial_norm = spatial_norm
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 3).uniform_()
        random_coords[:, 0] *= (self.cube_shape[0] - 1) / self.spatial_norm
        random_coords[:, 1] *= (self.cube_shape[1] - 1) / self.spatial_norm
        random_coords[:, 2] *= (self.cube_shape[2] - 1) / self.spatial_norm
        return random_coords