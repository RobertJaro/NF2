import numpy as np
from torch.utils.data import Dataset


class BoundaryDataset(Dataset):

    def __init__(self, coords, field, error, norm):
        self.norm = norm
        self.error = error
        self.coordinates = coords
        self.field = field

    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, idx):
        coord = self.coordinates[idx]
        field = self.field[idx]
        err = self.error[idx]
        scaled_coord = [c / self.norm for c in coord]
        return np.array([scaled_coord[0], scaled_coord[1], scaled_coord[2]], np.float32), field, err  # coord = (x, y, z)


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

    def __init__(self, cube_shape, norm, block_shape=(32, 32, 32), coords=[], strides=1):
        self.cube_shape = cube_shape
        self.block_shape = block_shape
        self.coords = coords
        self.strides = strides
        self.norm = norm

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return self.getCube(coord)

    def getCube(self, coord):
        coord_cube = np.stack(np.mgrid[coord[0]:coord[0] + self.block_shape[0]:self.strides,
                              coord[1]:coord[1] + self.block_shape[1]:self.strides,
                              coord[2]:coord[2] + self.block_shape[2]:self.strides, ], -1)
        coord_cube = np.stack([coord_cube[..., 0] / self.norm,
                               coord_cube[..., 1] / self.norm,
                               coord_cube[..., 2] / self.norm, ], -1)
        return coord_cube.astype(np.float32)
