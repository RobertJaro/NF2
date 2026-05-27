import glob
import os

import numpy as np
from astropy import units as u
from astropy.nddata import block_reduce
from tqdm import tqdm

from nf2.data.util import img_to_los_trv_azi
from nf2.loader.base import MapDataset

muram_variables = {'Bz': {'id': 'result_prim_5', 'unit': u.Gauss},
                   'By': {'id': 'result_prim_7', 'unit': u.Gauss},
                   'Bx': {'id': 'result_prim_6', 'unit': u.Gauss},
                   # 'vx': {'id': 'result_prim_1', 'unit': u.cm / u.s},
                   # 'vy': {'id': 'result_prim_2', 'unit': u.cm / u.s},
                   # 'vz': {'id': 'result_prim_3', 'unit': u.cm / u.s},
                   'tau': {'id': 'tau', 'unit': 1},
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

    def load_cube(self, resolution=0.192 * u.Mm / u.pix, height=100 * u.Mm, target_tau=1, method='median'):
        b = self.B
        tau = self.tau

        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        assert resolution % self.ds[2] == 0, f'resolution {resolution} must be a multiple of {self.ds[2]}'
        x_binning = resolution // self.ds[0]
        y_binning = resolution // self.ds[1]
        z_binning = resolution // self.ds[2]

        b = block_reduce(b, (x_binning, y_binning, z_binning, 1), np.mean)
        tau = block_reduce(tau, (x_binning, y_binning, z_binning), np.mean)

        pix_height = np.argmin(np.abs(tau - target_tau), axis=2) * u.pix
        if method == 'mean':
            base_height_pix = pix_height.mean()
        elif method == 'median':
            base_height_pix = np.median(pix_height)
        elif method == 'min':
            base_height_pix = np.min(pix_height)
        elif method == 'max':
            base_height_pix = np.max(pix_height)
        else:
            raise ValueError(f'Unknown method {method} for base height calculation')

        min_height = int(base_height_pix.to_value(u.pix))
        max_height = min_height + int((height / resolution).to_value(u.pix))
        b = b[:, :, min_height:max_height]
        tau = tau[:, :, min_height:max_height]

        return {'B': b, 'tau': tau}

    def load_tau(self, target_tau, **kwargs):
        tau = self.tau

        pix_height = np.argmin(np.abs(tau - target_tau), axis=2)
        height = int(pix_height.mean())

        return self.load_slice(**kwargs, height=height)

    def load_tau_height(self, target_tau, base_tau=1.0, method='mean'):
        tau = self.tau
        base_pix_height = np.argmin(np.abs(tau - base_tau), axis=2)
        if method == 'mean':
            base_pix_height = base_pix_height.mean()
        elif method == 'median':
            base_pix_height = np.median(base_pix_height)
        elif method == 'min':
            base_pix_height = np.min(base_pix_height)
        elif method == 'max':
            base_pix_height = np.max(base_pix_height)
        else:
            raise ValueError(f'Unknown method {method} for base height calculation')
        pix_height = np.argmin(np.abs(tau - target_tau), axis=2)
        return (pix_height - base_pix_height) * u.pix * self.ds[2]  # height in physical units

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

        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        assert resolution % self.ds[2] == 0, f'resolution {resolution} must be a multiple of {self.ds[2]}'
        x_binning = resolution // self.ds[0]
        y_binning = resolution // self.ds[1]
        z_binning = resolution // self.ds[2]

        b = block_reduce(b, (x_binning, y_binning, z_binning, 1), np.mean)
        tau = block_reduce(tau, (x_binning, y_binning, z_binning), np.mean)

        max_height = int((height / resolution).to_value(u.pix))
        b = b[:, :, :max_height]
        tau = tau[:, :, :max_height]

        return {'B': b, 'tau': tau}

    def get_tau_height_pix(self, tau, func=np.median, strides=16):
        # find closest to tau
        tau_cube = self.tau[::strides, ::strides, :]
        pix_height = np.argmin(np.abs(tau_cube - tau), axis=2)
        pix_height = func(pix_height, axis=(0, 1)).astype(int)
        return pix_height


def read_muram_slice(filepath):
    data = np.fromfile(filepath, dtype=np.float32)
    Nvar = data[0].astype(int)
    shape = tuple(data[1:3].astype(int))
    time = data[3]
    slice = data[4:].reshape([Nvar, shape[1], shape[0]]).swapaxes(1, 2)
    return slice, Nvar, shape, time


class MURaMSimulation:

    def __init__(self, source_path):
        files = sorted(glob.glob(os.path.join(source_path, 'Header.*')))
        # filter files --> check if tau files exists
        files = [f for f in files if os.path.exists(f.replace('Header', 'tau'))]
        iterations = [int(os.path.basename(f).split('.')[1]) for f in files]
        iterations.sort()

        snapshots = [MURaMSnapshot(source_path, i) for i in iterations]
        self.snapshots = {s.time.value: s for s in snapshots}
        self.iterations = {i: snapshots[idx] for idx, i in enumerate(iterations)}

    @property
    def times(self):
        return list(self.snapshots.keys())

    @property
    def ds(self):
        return list(self.snapshots.values())[0].ds

    def load_tau(self, tau=0.1, keys=('Bx', 'By', 'Bz'), spatial_strides=1):
        pix_height = self.get_average_height(tau)
        data = {k: [getattr(s, k)[::spatial_strides, ::spatial_strides, pix_height]
                    for s in self.snapshots.values()] for k in keys}
        data = {k: np.stack(v, -1) * muram_variables[k]['unit'] for k, v in tqdm(data.items())}
        data['time'] = np.array(list(self.snapshots.keys())) * u.s
        return data

    def get_average_height(self, tau):
        ref_snapshot = list(self.snapshots.values())[0]
        # find closest to tau = .1
        tau_cube = ref_snapshot.tau[::64, ::64, :]
        pix_height = np.argmin(np.abs(tau_cube - tau), axis=2)
        pix_height = np.mean(pix_height, axis=(0, 1)).astype(int)
        return pix_height

    def get_snapshot(self, time):
        return self.snapshots[time]
