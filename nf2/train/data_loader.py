import glob
import os
from copy import copy

import numpy as np
import wandb
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sunpy.map import Map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, BatchesDataset
from nf2.data.loader import prep_b_data, load_potential_field_data
from astropy import units as u

class SlicesDataModule(LightningDataModule):

    def __init__(self, b_slices, height, spatial_norm, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 error_slices=None, height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_strides = 1,
                 meta_data=None, plot_overview=True, Mm_per_pixel=None, **kwargs):
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
        assert len(height_mapping['z']) == b_slices.shape[2], 'Invalid height mapping configuration: z must have the same length as the number of slices'
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

        if boundary['type'] == 'potential':
            b_bottom = b_slices[:, :, 0]
            b_bottom = np.nan_to_num(b_bottom, nan=0) # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'], progress=True)
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
            b_bottom = np.nan_to_num(b_bottom, nan=0) # replace nans of mosaic data
            pf_coords, pf_errors, pf_values = load_potential_field_data(b_bottom, height, boundary['strides'], only_top=True, pf_error=0.1, progress=True)
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
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size, strides=validation_strides)
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


class SHARPDataModule(SlicesDataModule):

    def __init__(self, data_path, bin=1, slice=None, *args, **kwargs):
        if isinstance(data_path, str):
            hmi_p = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))[0]  # x
            hmi_t = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))[0]  # y
            hmi_r = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))[0]  # z
            err_p = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))[0]  # x
            err_t = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))[0]  # y
            err_r = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))[0]  # z
        else:
            hmi_p, err_p, hmi_t, err_t, hmi_r, err_r = data_path
        # laod maps
        p_map, t_map, r_map = Map(hmi_p),Map(hmi_t),Map(hmi_r)
        p_error_map, t_error_map, r_error_map = Map(err_p), Map(err_t), Map(err_r)

        maps = [p_map, t_map, r_map, p_error_map, t_error_map, r_error_map]
        if slice:
            maps = [m.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                             top_right=u.Quantity((slice[1], slice[3]), u.pixel)) for m in maps]
        if bin > 1:
            maps = [m.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean) for m in maps]

        hmi_data = np.stack([maps[0].data, -maps[1].data, maps[2].data]).transpose()
        error_data = np.stack([maps[3].data, maps[4].data, maps[5].data]).transpose()

        b_slices = hmi_data[:, :, None]
        error_slices = error_data[:, :, None]
        meta_data = maps[0].meta

        super().__init__(b_slices, *args, error_slices=error_slices, meta_data=meta_data, **kwargs)

class SOLISDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, *args, **kwargs):
        dict_data = np.load(data_path, allow_pickle=True)
        sharp_cube = dict_data.item().get('sharp')
        vsm_cube = dict_data.item().get('vsm')
        vsm_cube = np.stack([np.ones_like(vsm_cube) * np.nan, np.ones_like(vsm_cube) * np.nan, vsm_cube])
        b_slices = np.stack([sharp_cube, vsm_cube], 1).T
        if slices is not None:
            b_slices = b_slices[:, :, slices]

        super().__init__(b_slices, *args, **kwargs)


class FITSDataModule(SlicesDataModule):

    def __init__(self, fits_paths, mask_path=None, bin=1, *args, **kwargs):
        b_slices = []
        for f in fits_paths:
            b_slices += [fits.getdata(f).T]
        b_slices = np.stack(b_slices, 2)
        b_slices[..., 1] *= -1  # -t component

        if mask_path is not None:
            mask = fits.getdata(mask_path).T
            b_slices[mask == 0, :, :] = np.nan
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)

        meta_data = fits.getheader(fits_paths[0])

        super().__init__(b_slices, meta_data=meta_data, *args, **kwargs)


class SHARPSeriesDataModule(SHARPDataModule):

    def __init__(self, file_paths, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.file_paths = copy(file_paths)

        super().__init__(file_paths[0], *self.args, **self.kwargs)

    def train_dataloader(self):
        # re-initialize
        super().__init__(self.file_paths[0], *self.args, **self.kwargs)
        del self.file_paths[0] # continue with next file in list
        return super().train_dataloader()


class NumpyDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, bin=1, use_bz=False, *args, **kwargs):
        b_slices = np.load(data_path)
        if slices:
            b_slices = b_slices[:, :, slices]
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        if use_bz:
            b_slices[:, :, 1:, 0] = None
            b_slices[:, :, 1:, 1] = None
        super().__init__(b_slices, *args, **kwargs)


class AnalyticDataModule(LightningDataModule):

    def __init__(self, case, height, spatial_norm, b_norm, work_directory, batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None, boundary={"type": "full"}, **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm
        self.meta_data = None

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        tau_surfaces = boundary['tau_surfaces'] if boundary['type'] == 'tau' else None
        if case == 1:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi / 4, tau_surfaces=tau_surfaces)
        elif case == 2:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15, tau_surfaces=tau_surfaces,
                                          resolution=[80, 80, 72])
        else:
            raise Exception(f'Invalid CASE {case}. Available cases are: [1, 2]')

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(6, 2))
            min_max = np.abs(b_cube[..., i, 0]).max()
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')

        if boundary['type'] == "full":
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
        elif boundary['type'] == "potential":
            b_cube = b_cube[:, :, 0]
            coords, values, err = prep_b_data(b_cube, np.zeros_like(b_cube), height,
                                              potential_boundary=True,
                                              potential_strides=boundary['strides'])
        elif boundary['type'] == "open":
            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            height_slices = boundary['height_slices'] if 'height_slices' in boundary else 0
            b_cube = b_cube[:, :, height_slices]
            coords = coords[:, :, height_slices]
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
        elif boundary['type'] == "tau":

            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            ranges = np.zeros((*coords.shape[:3], 2), dtype=np.float32)

            height_mapping = {"z":  [h / 2 for h in tau_surfaces],
                                   "z_min": [0] * len(tau_surfaces),
                                   "z_max": [h for h in tau_surfaces]}
            self.height_mapping = height_mapping
            for i, (z, z_min, z_max) in enumerate(zip(height_mapping["z"], height_mapping["z_min"], height_mapping["z_max"])):
                ranges[:, :, i, 0] = z_min
                ranges[:, :, i, 1] = z_max
                coords[:, :, i, 2] = z
            #
            if boundary['use_LOS']:
                b_cube[:, :, 1:, 0] = None
                b_cube[:, :, 1:, 1] = None
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
            ranges = ranges.reshape((-1, 2)).astype(np.float32)
        else:
            raise Exception(f'Invalid boundary condition: {boundary}. Available options: ["full", "potential", "open"]')


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
        if boundary['type'] == "tau":  # add ranges
            ranges /= spatial_norm
            ranges = ranges[r]
            ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
            np.save(ranges_npy_path, ranges)
            batches_path['height_ranges'] = ranges_npy_path

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size)
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
