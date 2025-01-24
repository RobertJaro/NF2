import glob
import os
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.nddata import block_reduce
from tqdm import tqdm

muram_variables = {'Bz': {'id': 'result_prim_5', 'unit': u.Gauss},
                   'By': {'id': 'result_prim_7', 'unit': u.Gauss},
                   'Bx': {'id': 'result_prim_6', 'unit': u.Gauss},
                   'vz': {'id': 'result_prim_1', 'unit': u.cm / u.s},
                   'vx': {'id': 'result_prim_2', 'unit': u.cm / u.s},
                   'vy': {'id': 'result_prim_3', 'unit': u.cm / u.s},
                   'tau': {'id': 'tau', 'unit': 1},
                   'rho': {'id': 'result_prim_0', 'unit': u.g / u.cm ** 3},
                   'Pres': {'id': 'eosP', 'unit': u.erg / u.cm ** 3}
                   }


class MURaMSnapshot:

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
        self.iteration = iteration

    def get_B(self, binning=1):
        return np.stack([self.Bx[::binning, ::binning, ::binning],
                         self.By[::binning, ::binning, ::binning],
                         self.Bz[::binning, ::binning, ::binning]], axis=-1) * np.sqrt(4 * np.pi) * u.Gauss

    def get_v(self, binning=1):
        return np.stack([self.vx[::binning, ::binning, ::binning],
                         self.vy[::binning, ::binning, ::binning],
                         self.vz[::binning, ::binning, ::binning]], axis=-1) * u.cm / u.s

    def get_rho(self, binning=1):
        return self.rho[::binning, ::binning, ::binning] * u.g / u.cm ** 3

    def get_tau(self, binning=1):
        return self.tau[::binning, ::binning, ::binning]

    def load_cube(self, resolution=0.192 * u.Mm / u.pix, height=100 * u.Mm):
        b = self.B
        v = self.v
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
        v = block_reduce(v, (x_binning, y_binning, z_binning, 1), np.mean)

        pix_height = np.argmin(np.abs(tau - 1), axis=2) * u.pix
        base_height_pix = pix_height.mean()

        min_height = int(base_height_pix.to_value(u.pix))
        max_height = int((base_height_pix + height / resolution).to_value(u.pix))
        b = b[:, :, min_height:max_height]
        tau = tau[:, :, min_height:max_height]

        return {'B': b, 'v': v, 'tau': tau}

    def load_base(self, resolution=0.192 * u.Mm / u.pix, height=100 * u.Mm, base_height=132, rebin=False):
        # integer division
        assert resolution % self.ds[0] == 0, f'resolution {resolution} must be a multiple of {self.ds[0]}'
        assert resolution % self.ds[1] == 0, f'resolution {resolution} must be a multiple of {self.ds[1]}'
        assert resolution % self.ds[2] == 0, f'resolution {resolution} must be a multiple of {self.ds[2]}'
        x_binning = int(resolution // self.ds[0])
        y_binning = int(resolution // self.ds[1])
        z_binning = int(resolution // self.ds[2])

        max_height = int((height / self.ds[2]).to_value(u.pix)) + base_height
        if not rebin:
            bx = block_reduce(self.Bx[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            by = block_reduce(self.By[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            bz = block_reduce(self.Bz[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            b = np.stack([bx, by, bz], axis=-1) * np.sqrt(4 * np.pi) * u.Gauss

            vx = block_reduce(self.vx[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            vy = block_reduce(self.vy[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            vz = block_reduce(self.vz[:, :, base_height:max_height], (x_binning, y_binning, z_binning), np.mean)
            v = np.stack([vx, vy, vz], axis=-1) * u.cm / u.s

            rho = block_reduce(self.rho[:, :, base_height:max_height], (x_binning, y_binning, z_binning),
                               np.mean) * u.g / u.cm ** 3
            rho = rho[:, :, :, None]
        else:
            bx = self.Bx[::x_binning, ::y_binning, base_height:max_height:z_binning]
            by = self.By[::x_binning, ::y_binning, base_height:max_height:z_binning]
            bz = self.Bz[::x_binning, ::y_binning, base_height:max_height:z_binning]
            b = np.stack([bx, by, bz], axis=-1) * np.sqrt(4 * np.pi) * u.Gauss

            vx = self.vx[::x_binning, ::y_binning, base_height:max_height:z_binning]
            vy = self.vy[::x_binning, ::y_binning, base_height:max_height:z_binning]
            vz = self.vz[::x_binning, ::y_binning, base_height:max_height:z_binning]
            v = np.stack([vx, vy, vz], axis=-1) * u.cm / u.s

            rho = self.rho[::x_binning, ::y_binning, base_height:max_height:z_binning] * u.g / u.cm ** 3
            rho = rho[:, :, :, None]

        return {'B': b, 'v': v, 'rho': rho}


class MURaMSimulation():

    def __init__(self, source_path):
        files = sorted(glob.glob(os.path.join(source_path, 'Header.*')))
        # filter files --> check if tau files exists
        files = [f for f in files if os.path.exists(f.replace('Header', 'tau'))]
        iterations = [int(os.path.basename(f).split('.')[1]) for f in files]

        snapshots = [MURaMSnapshot(source_path, i) for i in iterations]
        self.snapshots = {s.time.value: s for s in snapshots}
        self.iterations = {i: s for i, s in zip(iterations, snapshots)}

    @property
    def times(self):
        return sorted(list(self.snapshots.keys()))

    @property
    def ds(self):
        return list(self.snapshots.values())[0].ds

    def load_tau(self, tau=0.1, keys=('Bx', 'By', 'Bz', 'vz'), spatial_strides=1):
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

    def get_iteration(self, iteration):
        return self.iterations[iteration]


class MURaMSimulationSlices():

    def __init__(self, data_path, time_sampling=12 * u.min):
        data_path = data_path if isinstance(data_path, list) else [data_path]
        files = [glob.glob(p) for p in data_path]
        files = np.concatenate(files)
        # sort by iteration number
        iteration = [int(os.path.basename(f).split('.')[-1]) for f in files]
        idx = np.argsort(iteration)
        files = files[idx]
        with Pool(16) as p:
            times = list(tqdm(p.imap(self._load_time, files), total=len(files)))
        times = np.array(times)  # in seconds
        # select times according to time_sampling
        selected_times = np.arange(times[0], times[-1], step=time_sampling.to(u.s).value)
        # find the closest file for each selected time
        selected_indices = [np.argmin(np.abs(times - t)) for t in selected_times]
        files = files[selected_indices]
        with Pool(16) as p:
            self.slices = list(tqdm(p.imap(MuramSlice, files), total=len(files)))

    def _load_time(self, file):
        data = np.memmap(file, mode='r', dtype=np.float32)
        time = data[3]
        return time

    @property
    def Bx(self):
        bx = np.stack([s.Bx for s in self.slices], axis=-1)
        return bx

    @property
    def By(self):
        by = np.stack([s.By for s in self.slices], axis=-1)
        return by

    @property
    def Bz(self):
        bz = np.stack([s.Bz for s in self.slices], axis=-1)
        return bz

    @property
    def vx(self):
        vx = np.stack([s.vx for s in self.slices], axis=-1)
        return vx

    @property
    def vy(self):
        vy = np.stack([s.vy for s in self.slices], axis=-1)
        return vy

    @property
    def vz(self):
        vz = np.stack([s.vz for s in self.slices], axis=-1)
        return vz

    @property
    def B(self):
        return np.stack([self.Bx, self.By, self.Bz], axis=-1)

    @property
    def v(self):
        return np.stack([self.vx, self.vy, self.vz], axis=-1)

    @property
    def time(self):
        return np.array([s.time.to(u.s).value for s in self.slices]) * u.s

    @property
    def rho(self):
        rho = np.stack([s.rho for s in self.slices], axis=-1)
        return rho

    @property
    def p(self):
        return np.stack([s.Pres for s in self.slices], axis=-1)


class MuramSlice():
    """2D slice through a MURaM domain

    This is a generic class that may represent horizontal, vertical, or tau slices.

    This object subclasses ndarray and may be used as if it were a normal NumPy array.
    The index ordering is [var, dim1, dim2]. The variable attributes (e.g. slice.rho) have
    dimension (dim1, dim2). The transpose option is for the xy_slice files that order the
    data by (var, y, x), which is counter-intuitive for something called "xy".

    Args:
      datapath (str): Path to a MURaM output directory containing slice files
      kind (str): Type of slice (xy, yz, or tau)
      ix (str): Index of the dimension orthogonal to the slice for (xy, yz), or tau value
      iteration (int): Iteration number
      transpose (bool): Whether to transpose the slice dimensions on instantiation.

    Attributes:
      ix (str): Index of the dimension orthogonal to the slice for (xy, yz), or tau value
      iteration (int): Iteration number
      time (float): Seconds since start of simulation
      rho (ndarray): density
      eint (ndarray): internal energy
      Temp (ndarray): temperature
      Pres (ndarray): pressure
      vx (ndarray): velocity in the x-direction
      vy (ndarray): velocity in the y-direction
      vz (ndarray): velocity in the z-direction
      Bx (ndarray): magnetic field in the x-direction
      By (ndarray): magnetic field in the y-direction
      Bz (ndarray): magnetic field in the z-direction
    """

    def __init__(self, file, transpose=False):
        sl, Nvar, shape, time = read_muram_slice(file)
        if transpose:
            sl = np.transpose(sl, [0, 2, 1])

        self.time = time * u.s
        self.rho = sl[0, :, :] * u.g / u.cm ** 3
        self.vz = sl[1, :, :] * u.cm / u.s
        self.vx = sl[2, :, :] * u.cm / u.s
        self.vy = sl[3, :, :] * u.cm / u.s
        self.eint = sl[4, :, :]
        self.Bz = sl[5, :, :] * np.sqrt(4 * np.pi) * u.Gauss
        self.Bx = sl[6, :, :] * np.sqrt(4 * np.pi) * u.Gauss
        self.By = sl[7, :, :] * np.sqrt(4 * np.pi) * u.Gauss
        if Nvar > 8:
            self.Temp = sl[8, :, :] * u.K
        if Nvar > 9:
            self.Pres = sl[9, :, :] * u.erg / u.cm ** 3


def read_muram_slice(filepath):
    data = np.fromfile(filepath, dtype=np.float32)
    Nvar = data[0].astype(int)
    shape = tuple(data[1:3].astype(int))
    time = data[3]
    slice = data[4:].reshape([Nvar, shape[1], shape[0]]).swapaxes(1, 2)
    return slice, Nvar, shape, time
