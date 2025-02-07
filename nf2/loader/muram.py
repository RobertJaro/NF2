import copy
import os

import numpy as np
import wandb
from astropy import units as u
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from nf2.data.dataset import RandomCoordinateDataset, CubeDataset, SlicesDataset
from nf2.data.util import img_to_los_trv_azi
from nf2.loader.base import MapDataset, BaseDataModule, TensorsDataset
from nf2.loader.fits import PotentialBoundaryDataset


class MURaMDataModule(BaseDataModule):

    def __init__(self, slices, work_directory, boundary_config=None, random_config=None,
                 Mm_per_ds=.36 * 320, G_per_dB=2500, seconds_per_dt=60, max_height=100,
                 batch_size=2**15, validation_batch_size=2 ** 15,
                 log_shape=False,
                 **kwargs):
        # boundary dataset
        slice_datasets = {}
        bottom_boundary_dataset = None
        slice_base_kwargs = {'G_per_dB': G_per_dB, 'Mm_per_ds': Mm_per_ds, 'seconds_per_dt': seconds_per_dt,
                             'work_directory': work_directory, 'batch_size': batch_size}
        for i, slice_config in enumerate(slices):
            slice_config = copy.deepcopy(slice_config)
            s_type = slice_config.pop('type', '2D')
            ds_id = slice_config.pop('name', f'boundary_{i + 1:02d}')
            bottom_boundary = slice_config.pop('bottom', False)
            slice_config = slice_base_kwargs | slice_config
            if s_type == 'slice':
                muram_dataset = MURaMDataset(**slice_config)
            elif s_type == 'cube':
                muram_dataset = MURaMCubeDataset(**slice_config)
            elif s_type == 'pressure':
                muram_dataset = MURaMPressureDataset(**slice_config)
            else:
                raise ValueError(f'Unknown slice type {s_type}')
            slice_datasets[ds_id] = muram_dataset
            if bottom_boundary:
                bottom_boundary_dataset = muram_dataset

        #
        if bottom_boundary_dataset is None:
            bottom_boundary_dataset = list(slice_datasets.values())[0]

        # random sampling dataset
        coord_range = bottom_boundary_dataset.coord_range
        z_range = np.array([[0, max_height / Mm_per_ds]])
        coord_range = np.concatenate([coord_range, z_range], axis=0)
        random_config = random_config if random_config is not None else {}
        random_dataset = RandomCoordinateDataset(coord_range, **random_config)

        ds_per_pixel = bottom_boundary_dataset.ds_per_pixel

        if log_shape:
            print(f'EXTRAPOLATING CUBE:')
            # pretty plot cube range
            print(f'x: {coord_range[0, 0] * Mm_per_ds:.2f} - {coord_range[0, 1] * Mm_per_ds:.2f} Mm')
            print(f'y: {coord_range[1, 0] * Mm_per_ds:.2f} - {coord_range[1, 1] * Mm_per_ds:.2f} Mm')
            print(f'z: {coord_range[2, 0] * Mm_per_ds:.2f} - {coord_range[2, 1] * Mm_per_ds:.2f} Mm')
            print('------------------')
            print(f'x: {coord_range[0, 0] / ds_per_pixel :.2f} - {coord_range[0, 1] / ds_per_pixel:.2f} pixel')
            print(f'y: {coord_range[1, 0] / ds_per_pixel:.2f} - {coord_range[1, 1] / ds_per_pixel:.2f} pixel')
            print(f'z: {coord_range[2, 0] / ds_per_pixel:.2f} - {coord_range[2, 1] / ds_per_pixel:.2f} pixel')
            print('------------------')
            print(f'x: {coord_range[0, 0]:.2f} - {coord_range[0, 1]:.2f} ds')
            print(f'y: {coord_range[1, 0]:.2f} - {coord_range[1, 1]:.2f} ds')
            print(f'z: {coord_range[2, 0]:.2f} - {coord_range[2, 1]:.2f} ds')

        training_datasets = {}
        for ds_id, dataset in slice_datasets.items():
            training_datasets[ds_id] = dataset
        training_datasets['random'] = random_dataset

        # top and side boundaries
        boundary_config = boundary_config if boundary_config is not None else {'type': 'potential', 'strides': 4}
        if boundary_config['type'] == 'potential':
            bz = bottom_boundary_dataset.bz
            potential_dataset = PotentialBoundaryDataset(bz=bz, height_pixel=max_height / (ds_per_pixel * Mm_per_ds),
                                                         ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                         work_directory=work_directory,
                                                         strides=boundary_config['strides'])
            training_datasets['potential'] = potential_dataset

        # validation datasets
        cube_dataset = CubeDataset(coord_range, batch_size=validation_batch_size)

        validation_slice_datasets = []
        slice_base_kwargs = {'G_per_dB': G_per_dB, 'Mm_per_ds': Mm_per_ds, 'seconds_per_dt': seconds_per_dt,
                             'work_directory': work_directory,
                             'shuffle': False, 'filter_nans': False, 'plot': False}
        for slice_config in slices:
            slice_config = copy.deepcopy(slice_config)
            s_type = slice_config.pop('type', '2D')
            slice_config['batch_size'] = validation_batch_size  # override batch size
            slice_config = slice_base_kwargs | slice_config
            if s_type == 'slice':
                muram_dataset = MURaMDataset(**slice_config)
            elif s_type == 'cube':
                muram_dataset = MURaMCubeDataset(**slice_config)
            elif s_type == 'pressure':
                muram_dataset = MURaMPressureDataset(**slice_config)
            else:
                raise ValueError(f'Unknown slice type {s_type}')
            validation_slice_datasets.append(muram_dataset)
        validation_slices_dataset = SlicesDataset(coord_range, ds_per_pixel, n_slices=10,
                                                  batch_size=validation_batch_size)

        validation_datasets = {'cube': cube_dataset, 'slices': validation_slices_dataset}
        for i, dataset in enumerate(validation_slice_datasets):
            validation_datasets[f'validation_boundary_{i + 1:02d}'] = dataset

        config = {'type': 'cartesian',
                  'Mm_per_ds': Mm_per_ds, 'G_per_dB': G_per_dB, 'seconds_per_dt': seconds_per_dt,
                  'max_height': max_height,
                  'coord_range': [], 'ds_per_pixel': [], 'height_mapping': []}
        for ds in slice_datasets.values():
            config['coord_range'].append(ds.coord_range)
            config['ds_per_pixel'].append(ds.ds_per_pixel)
            config['height_mapping'].append(ds.height_mapping)

        super().__init__(training_datasets, validation_datasets, config, **kwargs)


muram_variables = {'Bz': {'id': 'result_prim_5', 'unit': u.Gauss},
                   'By': {'id': 'result_prim_7', 'unit': u.Gauss},
                   'Bx': {'id': 'result_prim_6', 'unit': u.Gauss},
                   'vx': {'id': 'result_prim_1', 'unit': u.cm / u.s},
                   'vy': {'id': 'result_prim_2', 'unit': u.cm / u.s},
                   'vz': {'id': 'result_prim_3', 'unit': u.cm / u.s},
                   'tau': {'id': 'tau', 'unit': 1},
                   'P': {'id': 'eosP', 'unit': u.erg / u.cm ** 3},
                   'rho': {'id': 'result_prim_0', 'unit': u.g / u.cm ** 3},
                   }


class MURaMDataset(MapDataset):

    def __init__(self, data_path, los_trv_azi_transform=False, scaling=None, *args, **kwargs):
        sl, Nvar, shape, time = read_muram_slice(data_path)

        bz = sl[5, :, :] * np.sqrt(4 * np.pi)
        bx = sl[6, :, :] * np.sqrt(4 * np.pi)
        by = sl[7, :, :] * np.sqrt(4 * np.pi)

        b = np.stack([bx, by, bz], axis=-1)

        if scaling is not None:
            b = b * scaling

        if los_trv_azi_transform:
            b = img_to_los_trv_azi(b, f=np)

        super().__init__(b, Mm_per_pixel=0.192, los_trv_azi=los_trv_azi_transform, *args, **kwargs)


class MURaMCubeDataset(MapDataset):

    def __init__(self, data_path, iteration, base_height, *args, **kwargs):
        snapshot = MURaMSnapshot(data_path, iteration)

        b = snapshot.B
        b = b[:, :, base_height]

        super().__init__(b, Mm_per_pixel=0.192, *args, **kwargs)

class MURaMPressureDataset(TensorsDataset):
    def __init__(self, data_path, iteration, base_height, G_per_dB, Mm_per_ds, wcs=None, **kwargs):
        # Load MURaM snapshot
        snapshot = MURaMSnapshot(data_path, iteration)
        p = snapshot.P[:, :, base_height:, None]  # Trim height dimension

        # Normalize pressure
        p = p / G_per_dB ** 2

        # Extract spatial resolution
        dx, dy, dz = snapshot.ds
        # Define boundary coordinates
        x_dim, y_dim, z_dim = p.shape[:3]
        coords_boundary = self._generate_boundary_coords(x_dim, y_dim, z_dim, dx, dy, dz, Mm_per_ds)

        # Extract pressure values at boundaries
        p_boundary = self._extract_boundary_pressure(p)

        # Initialize class attributes
        self.coord_range = np.array([
            [coords_boundary[:, 0].min(), coords_boundary[:, 0].max()],
            [coords_boundary[:, 1].min(), coords_boundary[:, 1].max()]
        ])
        self.cube_shape = p.shape[:-1]
        self.wcs = wcs
        self.ds_per_pixel = dx.to_value(u.Mm / u.pix) / Mm_per_ds
        self.height_mapping = None

        # Create tensors dictionary
        tensors = {'p_true': p_boundary, 'coords': coords_boundary}

        # Call superclass constructor
        super().__init__(tensors, **kwargs)

        # Plotting (Optional)
        self._plot_pressure(p, G_per_dB)

    def _generate_boundary_coords(self, x_dim, y_dim, z_dim, dx, dy, dz, Mm_per_ds):
        """Generate coordinates for boundary planes in physical units."""

        # Create boundary planes
        coords_x_bottom = np.stack(np.meshgrid(0, np.arange(y_dim), np.arange(z_dim), indexing='ij'), axis=-1).reshape(
            -1, 3)
        coords_x_top = np.stack(np.meshgrid(x_dim - 1, np.arange(y_dim), np.arange(z_dim), indexing='ij'),
                                axis=-1).reshape(-1, 3)

        coords_y_bottom = np.stack(np.meshgrid(np.arange(x_dim), 0, np.arange(z_dim), indexing='ij'), axis=-1).reshape(
            -1, 3)
        coords_y_top = np.stack(np.meshgrid(np.arange(x_dim), y_dim - 1, np.arange(z_dim), indexing='ij'),
                                axis=-1).reshape(-1, 3)

        coords_z_bottom = np.stack(np.meshgrid(np.arange(x_dim), np.arange(y_dim), 0, indexing='ij'), axis=-1).reshape(
            -1, 3)
        coords_z_top = np.stack(np.meshgrid(np.arange(x_dim), np.arange(y_dim), z_dim - 1, indexing='ij'),
                                axis=-1).reshape(-1, 3)

        # Combine and remove duplicates
        boundary_coords = np.concatenate(
            [coords_x_bottom, coords_x_top, coords_y_bottom, coords_y_top, coords_z_bottom, coords_z_top],
            dtype=np.float32)

        # Scale to physical units
        boundary_coords[:, 0] *= dx.to_value(u.Mm / u.pix) / Mm_per_ds
        boundary_coords[:, 1] *= dy.to_value(u.Mm / u.pix) / Mm_per_ds
        boundary_coords[:, 2] *= dz.to_value(u.Mm / u.pix) / Mm_per_ds

        return boundary_coords

    def _extract_boundary_pressure(self, p):
        """Extract pressure values at the boundary planes."""
        return np.concatenate([
            p[0, :, :].reshape(-1, 1),  # x = 0
            p[-1, :, :].reshape(-1, 1),  # x = -1
            p[:, 0, :].reshape(-1, 1),  # y = 0
            p[:, -1, :].reshape(-1, 1),  # y = -1
            p[:, :, 0].reshape(-1, 1),  # z = 0
            p[:, :, -1].reshape(-1, 1)  # z = -1
        ])

    def _plot_pressure(self, p, G_per_dB):
        """Plot pressure and coordinate projections for verification."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot pressure slice at z=0
        im = ax.imshow(p[:, :, 0, 0].T * G_per_dB ** 2, origin='lower', cmap='viridis', norm=LogNorm())
        ax.set_title('Pressure at z=0')
        fig.colorbar(im, ax=ax, orientation='vertical')

        fig.tight_layout()
        wandb.log({'Pressure': wandb.Image(fig)})
        plt.close(fig)


class MURaMSnapshot():

    def __init__(self, source_path, iteration):
        header_file = os.path.join(source_path, f'Header.{iteration:06d}')
        assert os.path.exists(header_file), f'Header file {header_file} does not exist'

        # load shape information
        header = np.loadtxt(header_file, dtype=np.float32)
        shape = header[0:3].astype(int)
        ds = header[3:6] * u.cm / u.pix  # cm; pixel size
        # range = (shape * u.pix) * ds  # cm; domain size
        time = header[6] * u.s
        # (5, 6, 7) --> (z, x, y)

        files = {k: os.path.join(source_path, f"{v['id']}.{iteration:06d}") for k, v in muram_variables.items()}
        # (x, y, z) --> (z, y, x) --> (y, z, x)
        data = {k: np.memmap(f, mode='r', dtype=np.float32, shape=tuple(shape[::-1])).transpose(1, 0, 2)
                for k, f in files.items()}

        for k, v in data.items():
            setattr(self, k, v)
        self.time = time
        self.shape = (shape[1], shape[2], shape[0])
        self.ds = (ds[1], ds[2], ds[0])

    @property
    def B(self):
        return np.stack([self.Bx, self.By, self.Bz], axis=-1) * np.sqrt(4 * np.pi)

    def v(self):
        return np.stack([self.vx, self.vy, self.vz], axis=-1)

    def load_cube(self, resolution=0.192 * u.Mm / u.pix, height=100 * u.Mm, target_tau=1):
        b = self.B
        tau = self.tau
        p = self.P

        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        assert resolution % self.ds[2] == 0, f'resolution {resolution} must be a multiple of {self.ds[2]}'
        x_binning = resolution // self.ds[0]
        y_binning = resolution // self.ds[1]
        z_binning = resolution // self.ds[2]

        b = block_reduce(b, (x_binning, y_binning, z_binning, 1), np.mean)
        tau = block_reduce(tau, (x_binning, y_binning, z_binning), np.mean)
        p = block_reduce(p, (x_binning, y_binning, z_binning), np.mean)

        pix_height = np.argmin(np.abs(tau - target_tau), axis=2) * u.pix
        base_height_pix = pix_height.mean()

        min_height = int(base_height_pix.to_value(u.pix))
        max_height = min_height + int((height / resolution).to_value(u.pix))
        b = b[:, :, min_height:max_height]
        tau = tau[:, :, min_height:max_height]
        p = p[:, :, min_height:max_height]

        return {'B': b, 'tau': tau, 'P': p}

    def load_base(self, resolution=0.192 * u.Mm / u.pix, height=100 * u.Mm, base_height=180):
        b = self.B[:, :, base_height:]
        tau = self.tau[:, :, base_height:]
        p = self.P[:, :, base_height:]

        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        assert resolution % self.ds[2] == 0, f'resolution {resolution} must be a multiple of {self.ds[2]}'
        x_binning = resolution // self.ds[0]
        y_binning = resolution // self.ds[1]
        z_binning = resolution // self.ds[2]

        b = block_reduce(b, (x_binning, y_binning, z_binning, 1), np.mean)
        tau = block_reduce(tau, (x_binning, y_binning, z_binning), np.mean)
        p = block_reduce(p, (x_binning, y_binning, z_binning), np.mean)

        max_height = int((height / resolution).to_value(u.pix))
        b = b[:, :, :max_height]
        tau = tau[:, :, :max_height]
        p = p[:, :, :max_height]

        return {'B': b, 'tau': tau, 'P': p}


def read_muram_slice(filepath):
    data = np.fromfile(filepath, dtype=np.float32)
    Nvar = data[0].astype(int)
    shape = tuple(data[1:3].astype(int))
    time = data[3]
    slice = data[4:].reshape([Nvar, shape[1], shape[0]]).swapaxes(1, 2)
    return slice, Nvar, shape, time
