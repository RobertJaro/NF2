import os

import numpy as np
import wandb
from astropy import units as u
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from nf2.data.util import img_to_los_trv_azi
from nf2.loader.base import MapDataset, TensorsDataset

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

    def __init__(self, data_path, iteration, base_height=None, tau=None, *args, **kwargs):
        assert base_height is not None or tau is not None, 'Either base_height or tau must be provided'
        assert base_height is None or tau is None, 'Only one of base_height or tau can be provided'

        snapshot = MURaMSnapshot(data_path, iteration)
        if base_height is not None:
            b = snapshot.B
            b = b[:, :, base_height]
        elif tau is not None:
            muram_out = snapshot.load_tau(tau)
            b = muram_out['B']
        else:
            raise ValueError('Either base_height or tau must be provided')

        super().__init__(b, Mm_per_pixel=0.192, *args, **kwargs)


class MURaMPressureDataset(TensorsDataset):
    def __init__(self, data_path, iteration, G_per_dB, Mm_per_ds, slice_config=None, wcs=None, **kwargs):
        slice_config = slice_config if slice_config is not None else {'type': 'full', 'tau': 1}
        assert 'base_height' in slice_config or 'tau' in slice_config, 'Either base_height or tau must be provided in slice_config'


        # Load pressure cube from MURaM snapshot
        snapshot = MURaMSnapshot(data_path, iteration)
        if 'base_height' in slice_config:
            base_height = slice_config['base_height']
            p = snapshot.P
            p = p[:, :, base_height:, None]
        elif 'tau' in slice_config:
            tau = slice_config['tau']
            muram_out = snapshot.load_cube(target_tau=tau)
            p = muram_out['P'][..., None]
        else:
            raise ValueError('Either base_height or tau must be provided in slice_config')

        # Normalize pressure
        p = p / G_per_dB ** 2

        # Extract boundary or use full cube
        if slice_config['type'] == 'full':
            dx, dy, dz = snapshot.ds
            x_dim, y_dim, z_dim = p.shape[:3]
            coords = np.stack(np.meshgrid(np.arange(x_dim, dtype=np.float32),
                                          np.arange(y_dim, dtype=np.float32),
                                          np.arange(z_dim, dtype=np.float32), indexing='ij'), axis=-1)
            coords = coords.reshape(-1, 3)
            p = p.reshape(-1, 1)
        elif slice_config['type'] == 'boundary':
            # Extract spatial resolution
            dx, dy, dz = snapshot.ds
            # Define boundary coordinates
            x_dim, y_dim, z_dim = p.shape[:3]
            coords = self._generate_boundary_coords(x_dim, y_dim, z_dim)
            # Extract pressure values at boundaries
            p = self._extract_boundary_pressure(p)
        else:
            raise ValueError('Invalid slice type')

        # Scale to physical units
        coords[:, 0] *= dx.to_value(u.Mm / u.pix) / Mm_per_ds
        coords[:, 1] *= dy.to_value(u.Mm / u.pix) / Mm_per_ds
        coords[:, 2] *= dz.to_value(u.Mm / u.pix) / Mm_per_ds

        # Initialize class attributes
        self.coord_range = np.array([
            [coords[:, 0].min(), coords[:, 0].max()],
            [coords[:, 1].min(), coords[:, 1].max()]
        ])
        self.cube_shape = p.shape[:-1]
        self.wcs = wcs
        self.ds_per_pixel = dx.to_value(u.Mm / u.pix) / Mm_per_ds
        self.height_mapping = None

        # Create tensors dictionary
        tensors = {'p_true': p, 'coords': coords}

        # Call superclass constructor
        super().__init__(tensors, **kwargs)

        # Plotting (Optional)
        # self._plot_pressure(p, G_per_dB)

    def _generate_boundary_coords(self, x_dim, y_dim, z_dim):
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

    def load_tau(self, target_tau, **kwargs):
        tau = self.tau

        pix_height = np.argmin(np.abs(tau - target_tau), axis=2)
        height = int(pix_height.mean())

        return self.load_slice(**kwargs, height=height)

    def load_slice(self, height, resolution=0.192 * u.Mm / u.pix):
        b = self.B[:, :, height]
        p = self.P[:, :, height]

        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        x_binning = resolution // self.ds[0]
        y_binning = resolution // self.ds[1]

        b = block_reduce(b, (x_binning, y_binning, 1), np.mean)
        p = block_reduce(p, (x_binning, y_binning), np.mean)

        return {'B': b, 'P': p}


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
