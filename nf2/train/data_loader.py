import os

import numpy as np
from astropy.nddata import block_reduce
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BoundaryDataset, CubeDataset, RandomCoordinateDataset
from nf2.data.loader import load_hmi_data, prep_b_data, load_spherical_hmi_data


class SHARPDataModule(LightningDataModule):

    def __init__(self, data_path, height, spatial_norm, b_norm, work_directory, batch_size, random_batch_size,
                 iterations, num_workers,
                 use_potential_boundary=True, potential_strides=4, slice=None, bin=1):
        """
        :param b_cube: magnetic field data (x, y, (Bp, -Bt, Br)).
        :param error_cube: associated error information.
        :param height: height of simulation volume.
        :param spatial_norm: normalization of coordinate axis.
        :param b_norm: normalization of magnetic field strength.
        :param use_potential_boundary: use potential field as boundary condition. If None use an open boundary.
        :param potential_strides: use binned potential field boundary condition. Only applies if use_potential_boundary = True.
        """
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm

        # train parameters
        self.iterations = iterations
        self.num_workers = num_workers

        os.makedirs(work_directory, exist_ok=True)

        # prepare data
        b_cube, error_cube, meta_info = load_hmi_data(data_path)
        if slice:
            b_cube = b_cube[slice[0]:slice[1], slice[2]:slice[3]]
            error_cube = error_cube[slice[0]:slice[1], slice[2]:slice[3]]
        if bin > 1:
            b_cube = block_reduce(b_cube, (bin, bin, 1), np.mean)
            error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)

        self.b_cube = b_cube
        self.error_cube = error_cube
        self.meta_info = meta_info

        # load dataset
        data = prep_b_data(b_cube, error_cube, height, spatial_norm, b_norm,
                                plot=True, plot_path=work_directory,
                                potential_boundary=use_potential_boundary, potential_strides=potential_strides)
        cube_shape = [*b_cube.shape[:-1], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(data.shape[0])
        data = data[r]
        # adjust to batch size
        pad = batch_size - data.shape[0] % batch_size
        data = np.concatenate([data, data[:pad]])
        # split data into batches
        n_batches = data.shape[0] // batch_size
        batches = np.array(np.split(data, n_batches), dtype=np.float32)
        # store batches to disk
        batches_path = os.path.join(work_directory, 'batches.npy')
        np.save(batches_path, batches)
        # create data loaders
        self.dataset = BoundaryDataset(batches_path)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size)
        self.batches_path = batches_path

    def clear(self):
        os.remove(self.batches_path)

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return [cube_loader, boundary_loader]


class SynopticDataModule(LightningDataModule):

    def __init__(self, data_path, b_norm, height, work_directory, batch_size, iterations, num_workers, bin=1):
        """
        :param b_cube: magnetic field data (x, y, (Bp, -Bt, Br)).
        :param error_cube: associated error information.
        :param b_norm: normalization of magnetic field strength.
        :param use_potential_boundary: use potential field as boundary condition. If None use an open boundary.
        :param potential_strides: use binned potential field boundary condition. Only applies if use_potential_boundary = True.
        """
        super().__init__()

        # data parameters
        self.height = height
        self.b_norm = b_norm

        # train parameters
        self.iterations = iterations
        self.num_workers = num_workers

        os.makedirs(work_directory, exist_ok=True)

        # prepare data
        coords, b_cube, meta_info = load_spherical_hmi_data(data_path)
        if bin > 1:
            b_cube = block_reduce(b_cube, (bin, bin, 1), np.mean)
            coords = block_reduce(coords, (bin, bin, 1), np.mean)

        # load dataset
        b_cube = vector_spherical_to_cartesian(b_cube, coords)
        coords = to_cartesian(coords)
        b_cube /= b_norm
        error_cube = np.zeros_like(b_cube)

        self.b_cube = b_cube
        self.error_cube = error_cube
        self.meta_info = meta_info

        data = np.stack([b_cube.reshape((-1, 3)), error_cube.reshape((-1, 3)), coords.reshape((-1, 3))], axis=1)

        # prep dataset
        # shuffle data
        r = np.random.permutation(data.shape[0])
        data = data[r]
        # adjust to batch size
        pad = batch_size - data.shape[0] % batch_size
        data = np.concatenate([data, data[:pad]])
        # split data into batches
        n_batches = data.shape[0] // batch_size
        batches = np.array(np.split(data, n_batches), dtype=np.float32)
        # store batches to disk
        batches_path = os.path.join(work_directory, 'batches.npy')
        np.save(batches_path, batches)
        # create data loaders
        self.dataset = BoundaryDataset(batches_path)
        self.sphere_dataset = SphereDataset(height, batch_size=batch_size)
        self.batches_path = batches_path

    def clear(self):
        os.remove(self.batches_path)

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        return data_loader

    def val_dataloader(self):
        cube_loader = DataLoader(self.sphere_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return [cube_loader, boundary_loader]


def vector_spherical_to_cartesian(v, c):
    vp, vt, vr = v[:, 0], v[:, 1], v[:, 2]
    p, t, r = c[:, 0], c[:, 1], c[:, 2]
    sin = np.sin
    cos = np.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return np.stack([vx, vy, vz], -1)

def to_cartesian(v):
    sin = np.sin
    cos = np.cos
    p,t,r = v[..., 0], v[..., 1], v[..., 2]
    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return np.stack([x, y, z], -1)