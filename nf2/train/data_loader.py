import os

import numpy as np
import wandb
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset
from nf2.data.loader import load_hmi_data, prep_b_data, load_spherical_hmi_data, _plot_data, _load_potential_field_data
from nf2.data.analytical_field import get_analytic_b_field


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

        _plot_data(error_cube, b_cube, work_directory, b_norm)
        # load dataset
        coords, values, err = prep_b_data(b_cube, error_cube, height,
                                          potential_boundary=use_potential_boundary,
                                          potential_strides=potential_strides)
        # normalize data
        values = values / b_norm
        err = err / b_norm

        # apply spatial normalization
        coords = coords / spatial_norm

        cube_shape = [*b_cube.shape[:-1], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        err = err[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        err_npy_path = os.path.join(work_directory, 'err.npy')
        np.save(err_npy_path, err)
        # create data loaders
        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'errors': err_npy_path}
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size)
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
        return [cube_loader, boundary_loader]


class SyntheticMultiHeightDataModule(LightningDataModule):

    def __init__(self, data_path, height, spatial_norm, b_norm, work_directory, batch_size, random_batch_size,
                 iterations, num_workers, slice=None, bin=1,
                 use_potential_boundary=False, potential_strides=4):
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
        # Tau slices file
        # Bx, By, Bz, Babs: Gauss
        # mu (inclination), azimuth: degrees
        # dx, dy, dz, z_line: cm
        # tau_lev: no units
        # x is the vertical direction (64 km/pix)
        # y, z are in the horizontal plane (192 km/pix)
        # Dimensions: (nb_of_tau_levels, ny, nz)
        dict_data = dict(np.load(data_path))
        b_cube = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
        b_cube = np.moveaxis(b_cube, 0, -2)
        b_cube = block_reduce(b_cube, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution

        if slice:
            b_cube = b_cube[slice[0]:slice[1], slice[2]:slice[3], slice[4]:slice[5]]
        if bin > 1:
            b_cube = block_reduce(b_cube, (bin, bin, 1, 1), np.mean)

        # b_cube = b_cube[:, :, [0, -2]]
        # set x and y to None for upper layers
        # b_cube[:, :, 1:, 0] = None
        # b_cube[:, :, 1:, 1] = None

        self.b_cube = b_cube
        self.meta_info = None

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')
        # load dataset
        coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
        ranges = np.zeros((*coords.shape[:-1], 2))

        # coords[:, :, 0, 2] = 0
        # coords[:, :, 1, 2] = 3
        # ranges[:, :, 1, 1] = 50

        height_maps = dict_data['z_line'] / (dict_data['dy'] * 2) / bin # use spatial scaling of horizontal field
        height_maps -= 20 # shift 0 to photosphere
        # set first map fixed to photosphere
        height_maps[0, :, :] = 0
        # adjust for slices
        # height_maps = height_maps[:b_cube.shape[2]]
        average_heights = np.median(height_maps, axis=(1, 2))

        for i, h in enumerate(average_heights):
            coords[:, :, i, 2] = h

        max_heights = np.max(height_maps, axis=(1, 2))
        for i, h_max in enumerate(max_heights):
            ranges[:, :, i, 1] = h_max

        # flatten data
        coords = coords.reshape((-1, 3))
        values = b_cube.reshape((-1, 3))
        ranges = ranges.reshape((-1, 2))

        coords = coords.astype(np.float32)
        values = values.astype(np.float32)
        ranges = ranges.astype(np.float32)

        if use_potential_boundary:
            b_bottom = b_cube[:, :, 0]
            pf_coords, _, pf_values = _load_potential_field_data(b_bottom, height, potential_strides)
            #
            pf_ranges = np.zeros((*pf_coords.shape[:-1], 2), dtype=np.float32)
            pf_ranges[:, 0] = pf_coords[:, 2]
            pf_ranges[:, 1] = pf_coords[:, 2]
            # concatenate pf data points
            coords = np.concatenate([pf_coords, coords])
            values = np.concatenate([pf_values, values])
            ranges = np.concatenate([pf_ranges, ranges])

        # normalize data
        values = values / b_norm

        # apply spatial normalization
        coords = coords / spatial_norm
        ranges = ranges / spatial_norm

        cube_shape = [*b_cube.shape[:-2], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        ranges = ranges[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        np.save(ranges_npy_path, ranges)
        # create data loaders
        batches_path = {'coords': coords_npy_path,
                        'values':values_npy_path,
                        'height_ranges': ranges_npy_path}
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size)
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
        return [cube_loader, boundary_loader]


class AnalyticDataModule(LightningDataModule):

    def __init__(self, case, height_slices, height, spatial_norm, b_norm, work_directory, batch_size, random_batch_size,
                 iterations, num_workers, boundary="full", potential_strides=4, tau_surfaces=None, use_LOS=True):
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm

        # train parameters
        self.iterations = iterations
        self.num_workers = num_workers

        os.makedirs(work_directory, exist_ok=True)

        if case == 1:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi / 4, tau_surfaces=tau_surfaces)
        elif case == 2:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15, tau_surfaces=tau_surfaces, resolution=[80, 80, 72])
        else:
            raise Exception(f'Invalid CASE {case}. Available cases are: [1, 2]')

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            min_max = np.abs(b_cube[..., i, 0]).max()
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')

        if boundary == "full":
            coord_cube = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1)
            values = [b_cube[:, :, :1].reshape((-1, 3)), b_cube[:, :, -1:].reshape((-1, 3)),
                      b_cube[:, :1, :].reshape((-1, 3)), b_cube[:, -1:, :].reshape((-1, 3)),
                      b_cube[:1, :, :].reshape((-1, 3)), b_cube[-1:, :, :].reshape((-1, 3)), ]
            coords = [coord_cube[:, :, :1].reshape((-1, 3)), coord_cube[:, :, -1:].reshape((-1, 3)),
                      coord_cube[:, :1, :].reshape((-1, 3)), coord_cube[:, -1:, :].reshape((-1, 3)),
                      coord_cube[:1, :, :].reshape((-1, 3)), coord_cube[-1:, :, :].reshape((-1, 3)), ]
            #
            coords = np.concatenate(coords).astype(np.float32)
            values = np.concatenate(values).astype(np.float32)
        elif boundary == "potential":
            b_cube = b_cube[:, :, 0]
            coords, values, err = prep_b_data(b_cube, np.zeros_like(b_cube), height,
                                              potential_boundary=True,
                                              potential_strides=potential_strides)
        elif boundary == "open":
            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            b_cube = b_cube[:, :, height_slices]
            coords = coords[:, :, height_slices]
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
        elif boundary == "tau":
            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            ranges = np.zeros((*coords.shape[:3], 2), dtype=np.float32)
            for i, h in enumerate(tau_surfaces):
                coords[:, :, i, 2] = h / 2 # set to center of tau surface
                ranges[:, :, i, 1] = h
            #
            if use_LOS:
                b_cube[:, :, 1:, 0] = None
                b_cube[:, :, 1:, 1] = None
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
            ranges = ranges.reshape((-1, 2)).astype(np.float32)
        else:
            raise Exception(f'Invalid boundary condition: {boundary}. Available options: ["full", "potential", "open"]')

        self.meta_info = None

        # normalize data
        values = values / b_norm
        # apply spatial normalization
        coords = coords / spatial_norm

        cube_shape = [*b_cube.shape[:2], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        # create data loaders
        batches_path = {'coords': coords_npy_path, 'values': values_npy_path}
        if boundary == "tau": # add ranges
            ranges /= spatial_norm
            ranges = ranges[r]
            ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
            np.save(ranges_npy_path, ranges)
            batches_path['height_ranges'] = ranges_npy_path
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size)
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
        return [cube_loader, boundary_loader]

class VSMMultiHeightDataModule(LightningDataModule):

    def __init__(self, data_path, height_mapping, max_height, spatial_norm, b_norm, work_directory, batch_size, random_batch_size,
                 iterations, num_workers, slice=None, bin=1, return_height_ranges=True,
                 use_potential_boundary=False, potential_strides=4):
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
        self.height = max_height
        self.b_norm = b_norm

        # train parameters
        self.iterations = iterations
        self.num_workers = num_workers

        os.makedirs(work_directory, exist_ok=True)

        # prepare data
        # Tau slices file
        # Bx, By, Bz, Babs: Gauss
        # mu (inclination), azimuth: degrees
        # dx, dy, dz, z_line: cm
        # tau_lev: no units
        # x is the vertical direction (64 km/pix)
        # y, z are in the horizontal plane (192 km/pix)
        # Dimensions: (nb_of_tau_levels, ny, nz)
        dict_data = np.load(data_path, allow_pickle=True)
        sharp_cube = dict_data.item().get('sharp')
        vsm_cube = dict_data.item().get('vsm')
        vsm_cube = np.stack([np.ones_like(vsm_cube) * np.nan, np.ones_like(vsm_cube) * np.nan, vsm_cube])
        b_cube = np.stack([sharp_cube, vsm_cube], 1).T

        if slice:
            b_cube = b_cube[slice[0]:slice[1], slice[2]:slice[3], slice[4]:slice[5]]
        if bin > 1:
            b_cube = block_reduce(b_cube, (bin, bin, 1, 1), np.mean)

        self.b_cube = b_cube
        self.meta_info = None

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')
        # load dataset
        coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
        ranges = np.zeros((*coords.shape[:-1], 2))
        for i, (h, min_h, max_h) in enumerate(height_mapping):
            coords[:, :, i, 2] = h
            ranges[:, :, i, 0] = min_h
            ranges[:, :, i, 1] = max_h
        # flatten data
        coords = coords.reshape((-1, 3))
        values = b_cube.reshape((-1, 3))
        ranges = ranges.reshape((-1, 2))

        coords = coords.astype(np.float32)
        values = values.astype(np.float32)
        ranges = ranges.astype(np.float32)

        if use_potential_boundary:
            b_bottom = b_cube[:, :, 0]
            pf_coords, _, pf_values = _load_potential_field_data(b_bottom, max_height, potential_strides)
            #
            pf_ranges = np.zeros((*pf_coords.shape[:-1], 2), dtype=np.float32)
            # concatenate pf data points
            coords = np.concatenate([pf_coords, coords])
            values = np.concatenate([pf_values, values])
            ranges = np.concatenate([pf_ranges, ranges])

        # normalize data
        values = values / b_norm

        # apply spatial normalization
        coords /= spatial_norm
        ranges /= spatial_norm

        cube_shape = [*b_cube.shape[:-2], max_height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        ranges = ranges[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        np.save(ranges_npy_path, ranges)
        # create data loaders
        batches_path = {'coords': coords_npy_path,
                        'values':values_npy_path}
        if return_height_ranges:
            batches_path['height_ranges'] = ranges_npy_path
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size)
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
        return [cube_loader, boundary_loader]

class SSTDataModule(LightningDataModule):

    def __init__(self, fits_paths, height_mapping, max_height, spatial_norm, b_norm, work_directory, batch_size, random_batch_size,
                 iterations, num_workers, mask_path=None, slice=None, bin=1,
                 use_potential_boundary=False, potential_strides=4):
        """
        :param fits_paths: path to fits files.
        :param height_mapping: estimated height and allowed height ranges in pixels [(height, min_height, max_height), (...)].
        :param max_height: height of simulation volume in pixels.
        :param spatial_norm: normalization of coordinate axis.
        :param b_norm: normalization of magnetic field strength.
        :param use_potential_boundary: use potential field as boundary condition. If None use an open boundary.
        :param potential_strides: use binned potential field boundary condition. Only applies if use_potential_boundary = True.
        """
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = max_height
        self.b_norm = b_norm

        # train parameters
        self.iterations = iterations
        self.num_workers = num_workers

        os.makedirs(work_directory, exist_ok=True)

        # prepare data
        # Tau slices file
        # Bx, By, Bz, Babs: Gauss
        # mu (inclination), azimuth: degrees
        # dx, dy, dz, z_line: cm
        # tau_lev: no units
        # x is the vertical direction (64 km/pix)
        # y, z are in the horizontal plane (192 km/pix)
        # Dimensions: (nb_of_tau_levels, ny, nz)
        b_cube = []
        for f in fits_paths:
            b_cube += [fits.getdata(f).T]
        b_cube = np.stack(b_cube, 2)
        b_cube[..., 1] *= -1 # fix

        if mask_path is not None:
            mask = fits.getdata(mask_path).T
            b_cube[mask == 0, :, :] = np.nan
        if slice:
            b_cube = b_cube[slice[0]:slice[1], slice[2]:slice[3], slice[4]:slice[5]]
        if bin > 1:
            b_cube = block_reduce(b_cube, (bin, bin, 1, 1), np.mean)

        self.b_cube = b_cube
        self.meta_info = None

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')
        # load dataset
        coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
        ranges = np.zeros((*coords.shape[:-1], 2))
        for i, (h, min_h, max_h) in enumerate(height_mapping):
            coords[:, :, i, 2] = h
            ranges[:, :, i, 0] = min_h
            ranges[:, :, i, 1] = max_h

        # flatten data
        coords = coords.reshape((-1, 3))
        values = b_cube.reshape((-1, 3))
        ranges = ranges.reshape((-1, 2))

        coords = coords.astype(np.float32)
        values = values.astype(np.float32)
        ranges = ranges.astype(np.float32)

        if use_potential_boundary:
            b_bottom = b_cube[:, :, 0]
            pf_coords, _, pf_values = _load_potential_field_data(b_bottom, max_height, potential_strides, only_top=True)
            #
            pf_ranges = np.ones((*pf_coords.shape[:-1], 2), dtype=np.float32)
            pf_ranges[..., 0] = height_mapping[-1][2] # min at last height
            pf_ranges[..., 1] = max_height # max at top
            # concatenate pf data points
            coords = np.concatenate([pf_coords, coords])
            values = np.concatenate([pf_values, values])
            ranges = np.concatenate([pf_ranges, ranges])

        # normalize data
        values = values / b_norm

        # apply spatial normalization
        coords /= spatial_norm
        ranges /= spatial_norm

        cube_shape = [*b_cube.shape[:-2], max_height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        ranges = ranges[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        np.save(ranges_npy_path, ranges)
        # create data loaders
        batches_path = {'coords': coords_npy_path,
                        'values':values_npy_path,
                        'height_ranges': ranges_npy_path}
        self.dataset = BatchesDataset(batches_path, batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=batch_size, strides=8)
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
    p, t, r = v[..., 0], v[..., 1], v[..., 2]
    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return np.stack([x, y, z], -1)
