import copy
import glob
import os
from copy import deepcopy

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from pytorch_lightning import LightningDataModule
from sunpy.map import Map

from nf2.data.dataset import RandomCoordinateDataset, CubeDataset, SlicesDataset, RandomHeightCoordinateDataset
from nf2.data.loader import load_potential_field_boundary
from nf2.loader.analytical import AnalyticalDataset
from nf2.loader.base import TensorsDataset, BaseDataModule, MapDataset
from nf2.loader.muram import MURaMDataset, MURaMCubeDataset, MURaMPressureDataset


class CartesianDataModule(BaseDataModule):

    def __init__(self, train_configs, work_directory, valid_configs=None,
                 boundary_config=None, random_config=None,
                 Mm_per_ds=.36 * 320, G_per_dB=2500, z_range=None, validation_batch_size=2 ** 14, log_shape=False,
                 batch_size=2 ** 13, validation_pixel_per_ds=128, iterations=None,
                 num_workers=None, **kwargs):
        self.Mm_per_ds = Mm_per_ds
        self.G_per_dB = G_per_dB
        self.work_directory = work_directory
        self.batch_size = batch_size
        self.ds_batch_size = batch_size // len(train_configs)
        self.validation_batch_size = validation_batch_size
        self.validation_pixel_per_ds = validation_pixel_per_ds
        # wrap data if only one slice is provided
        train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
        # boundary dataset
        num_workers = num_workers if num_workers is not None else os.cpu_count()
        training_datasets = [self._load_ds_config(config) for config in train_configs]
        training_datasets = dict(training_datasets)  # convert to dict

        bottom_boundary_dataset = list(training_datasets.values())[0]

        # random sampling dataset
        coord_range = bottom_boundary_dataset.coord_range
        z_range = [0, 100] if z_range is None else z_range
        z_range_arr = np.array([z_range]) / Mm_per_ds
        coord_range = np.concatenate([coord_range, z_range_arr], axis=0)
        random_config = copy.deepcopy(random_config) if random_config is not None else {'batch_size': 2 ** 14}
        random_type = random_config.pop('type', 'default')
        if random_type == 'default':
            random_dataset = RandomCoordinateDataset(coord_range, length=iterations, **random_config)
        elif random_type == 'height':
            random_dataset = RandomHeightCoordinateDataset(coord_range, length=iterations, **random_config)
        else:
            raise ValueError(f'Unknown random dataset type: {random_type}')

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

        self.coord_range = coord_range
        training_datasets['random'] = random_dataset

        # top and side boundaries
        boundary_config = boundary_config if boundary_config is not None else \
            {'type': 'potential', 'strides': 4}
        boundary_config = deepcopy(boundary_config)
        boundary_batch_size = boundary_config.pop('batch_size', batch_size // 4)
        boundary_type = boundary_config.pop('type')
        if boundary_type == 'none':
            pass
        elif boundary_type == 'potential':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            boundary_ds = PotentialBoundaryDataset(bz=bz,
                                                   height_pixel=coord_range[2, -1] / ds_per_pixel,
                                                   coord_range=coord_range,
                                                   ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                   work_directory=work_directory,
                                                   batch_size=boundary_batch_size, **boundary_config)
            training_datasets['potential'] = boundary_ds
        elif boundary_type == 'potential_top':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            boundary_ds = PotentialTopBoundaryDataset(bz=bz,
                                                      height_pixel=coord_range[2, -1] / ds_per_pixel,
                                                      coord_range=coord_range,
                                                      ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                      work_directory=work_directory,
                                                      batch_size=boundary_batch_size, **boundary_config)
            training_datasets['potential'] = boundary_ds
        else:
            raise ValueError(f'Unknown boundary type: {boundary_config["type"]}')

        # validation datasets
        validation_datasets = {}
        if valid_configs is None:
            valid_configs = train_configs
            valid_configs.append({'type': "cube", 'ds_id': 'cube', })
            valid_configs.append({'type': "slices", 'ds_id': 'slices', })

        validation_datasets = [self._load_valid_ds_config(config) for config in valid_configs]
        validation_datasets = dict(validation_datasets)  # convert to dict

        config = {'type': 'cartesian', 'max_height': z_range[1],
                  'Mm_per_ds': Mm_per_ds, 'G_per_dB': G_per_dB,
                  'coord_range': [], 'ds_per_pixel': [], 'height_mapping': [], 'wcs': [],
                  'cube_shape': []}
        for dataset in training_datasets.values():
            if not isinstance(dataset, MapDataset):
                continue
            config['coord_range'].append(dataset.coord_range)
            config['cube_shape'].append(dataset.cube_shape)
            config['ds_per_pixel'].append(dataset.ds_per_pixel)
            config['height_mapping'].append(dataset.height_mapping)
            config['wcs'].append(dataset.wcs)

        super().__init__(training_datasets, validation_datasets, config, num_workers=num_workers, **kwargs)

    def init_dataset(self, type, **kwargs):
        if type == 'fits':
            return FITSDataset(**kwargs)
        elif type == 'los_trv_azi':
            return LosTrvAziFITSDataset(**kwargs)
        elif type == 'los':
            return LosFITSDataset(**kwargs)
        elif type == 'sharp':
            return SHARPDataset(**kwargs)
        elif type == 'fld_inc_azi':
            return FldIncAziFITSDataset(**kwargs)
        elif type == 'numpy':
            return NumpyDataset(**kwargs)
        elif type == 'muram_slice':
            return MURaMDataset(**kwargs)
        elif type == 'muram_cube':
            return MURaMCubeDataset(**kwargs)
        elif type == 'muram_pressure':
            return MURaMPressureDataset(**kwargs)
        elif type == 'analytical':
            return AnalyticalDataset(**kwargs)
        elif type == 'slices':
            return SlicesDataset(**kwargs)
        elif type == 'cube':
            return CubeDataset(**kwargs)
        else:
            raise ValueError(f'Unknown boundary type: {type}. Supported types: '
                             f'fits, los_trv_azi, los, sharp, fld_inc_azi, numpy, muram_slice, muram_cube, '
                             f'muram_pressure, analytical, slices, cube')

    def _load_ds_config(self, config):
        """
        Load dataset from configuration.
        :param config: dataset configuration
        :return: dataset instance
        """
        config = deepcopy(config)
        ds_type = config['type']
        ds_id = config.pop('ds_id', f'train_{ds_type}')
        config['batch_size'] = config.pop('batch_size', self.ds_batch_size)  # default batch size
        dataset = self.init_dataset(**config, Mm_per_ds=self.Mm_per_ds, G_per_dB=self.G_per_dB,
                                    work_directory=self.work_directory)
        return ds_id, dataset

    def _load_valid_ds_config(self, config):
        """
        Load validation dataset from configuration.
        :param config: dataset configuration
        :return: dataset instance
        """
        config = deepcopy(config)
        ds_type = config['type']
        ds_id = config.pop('ds_id', f'valid_{ds_type}')
        config['batch_size'] = config.pop('batch_size', self.validation_batch_size)
        plot = config.pop('plot', False)
        dataset = self.init_dataset(**config,
                                    Mm_per_ds=self.Mm_per_ds, G_per_dB=self.G_per_dB,
                                    shuffle=False, filter_nans=False,
                                    work_directory=self.work_directory, plot=plot,
                                    ds_per_pixel=1 / self.validation_pixel_per_ds, coord_range=self.coord_range)
        return ds_id, dataset


class FITSDataset(MapDataset):

    def __init__(self, fits_path, error_path=None, coords_path=None,
                 bin=1, slice=None, load_map=True, **kwargs):
        file_p = fits_path['Bp']
        file_t = fits_path['Bt']
        file_r = fits_path['Br']

        if load_map:
            p_map, t_map, r_map = Map(file_p), Map(file_t), Map(file_r)
            p_map = process_map(p_map, slice, bin)
            t_map = process_map(t_map, slice, bin)
            r_map = process_map(r_map, slice, bin)

            b = np.stack([p_map.data, -t_map.data, r_map.data]).transpose()
            wcs = r_map.wcs
        else:
            p_data = fits.getdata(file_p)
            t_data = fits.getdata(file_t)
            r_data = fits.getdata(file_r)
            if slice:
                p_data = p_data[slice[0]:slice[1], slice[2]:slice[3]]
                t_data = t_data[slice[0]:slice[1], slice[2]:slice[3]]
                r_data = r_data[slice[0]:slice[1], slice[2]:slice[3]]
            if bin > 1:
                p_data = block_reduce(p_data, (bin, bin), func=np.mean)
                t_data = block_reduce(t_data, (bin, bin), func=np.mean)
                r_data = block_reduce(r_data, (bin, bin), func=np.mean)

            b = np.stack([p_data, -t_data, r_data]).transpose()
            wcs = None

        if error_path is not None:
            if isinstance(error_path, str):
                file_p_err = sorted(glob.glob(os.path.join(error_path, '*Bp_err.fits')))[0]  # x
                file_t_err = sorted(glob.glob(os.path.join(error_path, '*Bt_err.fits')))[0]  # y
                file_r_err = sorted(glob.glob(os.path.join(error_path, '*Br_err.fits')))[0]  # z
            else:
                file_p_err = error_path['Bp_err']
                file_t_err = error_path['Bt_err']
                file_r_err = error_path['Br_err']
            p_error_map, t_error_map, r_error_map = Map(file_p_err), Map(file_t_err), Map(file_r_err)
            p_error_map = process_map(p_error_map, slice, bin)
            t_error_map = process_map(t_error_map, slice, bin)
            r_error_map = process_map(r_error_map, slice, bin)
            b_err = np.stack([p_error_map.data, t_error_map.data, r_error_map.data]).transpose()
        else:
            b_err = None

        if coords_path is not None:
            coords_x, coords_y, coords_z = Map(coords_path['x']), Map(coords_path['y']), Map(coords_path['z'])
            coords_x = process_map(coords_x, slice, bin)
            coords_y = process_map(coords_y, slice, bin)
            coords_z = process_map(coords_z, slice, bin)
            coords = np.stack([coords_x.data, coords_y.data, coords_z.data]).transpose()
        else:
            coords = None

        super().__init__(b=b, b_err=b_err, coords=coords, wcs=wcs, **kwargs)


class LosTrvAziFITSDataset(MapDataset):

    def __init__(self, fits_path, mask_config=None,
                 bin=1, slice=None, load_map=True, Mm_per_pixel=0.36, **kwargs):
        file_los = fits_path['B_los']
        file_trv = fits_path['B_trv']
        file_azi = fits_path['B_azi']
        file_z = fits_path.get('Z', None)

        if load_map:
            los_map, trv_map, azi_map = Map(file_los), Map(file_trv), Map(file_azi)
            los_map = process_map(los_map, slice, bin)
            trv_map = process_map(trv_map, slice, bin)
            azi_map = process_map(azi_map, slice, bin)
            los_data = los_map.data
            trv_data = trv_map.data
            azi_data = azi_map.data
            wcs = los_map.wcs
            if mask_config is not None:
                mask_map = Map(mask_config['path'])
                mask_map = process_map(mask_map, slice, bin)
                mask = mask_map.data
                los_data[mask] = 0  # np.nan
                trv_data[mask] = 0  # np.nan
                azi_data[mask] = 0  # np.nan
        else:
            los_data = fits.getdata(file_los)
            trv_data = fits.getdata(file_trv)
            azi_data = fits.getdata(file_azi)
            if file_z is not None:
                bunit = fits.getheader(file_z)['BUNIT']
                z_data = fits.getdata(file_z) * u.Quantity(1, bunit)
            else:
                z_data = np.zeros_like(los_data)
            if mask_config is not None:
                mask = fits.getdata(mask_config['path'])
                mask = np.array(mask, dtype=bool)
                los_data[mask] = mask_config.get('value', np.nan)
                trv_data[mask] = mask_config.get('value', np.nan)
                azi_data[mask] = mask_config.get('value', np.nan)
                z_data[mask] = mask_config.get('value', np.nan)
            if slice:
                los_data = los_data[slice[0]:slice[1], slice[2]:slice[3]]
                trv_data = trv_data[slice[0]:slice[1], slice[2]:slice[3]]
                azi_data = azi_data[slice[0]:slice[1], slice[2]:slice[3]]
                z_data = z_data[slice[0]:slice[1], slice[2]:slice[3]]
            if bin > 1:
                los_data = block_reduce(los_data, (bin, bin), func=np.mean)
                trv_data = block_reduce(trv_data, (bin, bin), func=np.mean)
                azi_data = block_reduce(azi_data, (bin, bin), func=np.mean)
                z_data = block_reduce(z_data, (bin, bin), func=np.mean)
            wcs = None

        b = np.stack([los_data, trv_data, np.pi - azi_data]).transpose()

        coords = np.stack(np.mgrid[:b.shape[0], :b.shape[1], :1], -1).astype(np.float32) * Mm_per_pixel
        coords = coords[:, :, 0, :]
        if file_z is not None:
            coords[..., 2] = z_data.T.to_value(u.Mm)
        else:
            coords = None

        super().__init__(b=b, wcs=wcs, los_trv_azi=True, coords=coords, Mm_per_pixel=Mm_per_pixel, **kwargs)


class LosFITSDataset(MapDataset):

    def __init__(self, fits_path, mask_path=None, mask_value=np.nan,
                 bin=1, slice=None, load_map=True, **kwargs):
        file_los = fits_path['B_los']

        if load_map:
            los_map = Map(file_los)
            if mask_path is not None:
                mask_map = Map(mask_path)
                mask = mask_map.data
                los_map.data[mask] = mask_value
            los_map = process_map(los_map, slice, bin)
            los_data = los_map.data
            wcs = los_map.wcs
        else:
            los_data = fits.getdata(file_los)
            if mask_path is not None:
                mask = fits.getdata(mask_path)
                mask = np.array(mask, dtype=bool)
                los_data[mask] = mask_value
            if slice:
                los_data = los_data[slice[0]:slice[1], slice[2]:slice[3]]
            if bin > 1:
                los_data = block_reduce(los_data, (bin, bin), func=np.mean)
            wcs = None

        B_nan = np.ones_like(los_data) * np.nan
        b = np.stack([B_nan, B_nan, los_data]).transpose()

        super().__init__(b=b, wcs=wcs, **kwargs)


class FldIncAziFITSDataset(MapDataset):

    def __init__(self, fits_path, bin=1, slice=None, **kwargs):
        file_fld = fits_path['B_fld']
        file_inc = fits_path['B_inc']
        file_azi = fits_path['B_azi']

        fld_map, inc_map, azi_map = Map(file_fld), Map(file_inc), Map(file_azi)
        fld_map.data[:] = np.flip(fld_map.data, axis=(0, 1))
        inc_map.data[:] = np.flip(inc_map.data, axis=(0, 1))
        azi_map.data[:] = np.flip(azi_map.data, axis=(0, 1))

        fld_map = process_map(fld_map, slice, bin)
        inc_map = process_map(inc_map, slice, bin)
        azi_map = process_map(azi_map, slice, bin)

        fld = fld_map.data
        inc = np.deg2rad(inc_map.data)
        azi = np.deg2rad(azi_map.data)

        # apply disambiguation
        if 'B_amb' in fits_path:
            file_amb = fits_path['B_amb']
            amb_map = Map(file_amb)
            amb_map.data[:] = np.flip(amb_map.data, axis=(0, 1))
            amb_map = process_map(amb_map, slice, bin)
            amb = amb_map.data

            amb_weak = 2
            condition = (amb.astype(int) >> amb_weak).astype(bool)
            azi[condition] += np.pi

        wcs = fld_map.wcs

        los = fld * np.cos(inc)
        trv = fld * np.sin(inc)

        b = np.stack([los, trv, (np.pi - azi) % (2 * np.pi)]).transpose()

        super().__init__(b=b, wcs=wcs, los_trv_azi=True, **kwargs)


class CartesianSeriesDataModule(LightningDataModule):

    def __init__(self, train_configs, current_step, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        train_configs = [_load_config(c) for c in train_configs]
        assert all([len(c) == len(train_configs[0]) for c in train_configs]), \
            'Inconsistent number of training files in configurations. Check your configurations.'
        train_configs = list(zip(*train_configs))

        # remove already extrapolated files
        assert current_step < len(train_configs), \
            'Not enough data files found to continue training. Training completed or configuration is incorrect.'

        self.step = current_step
        self.train_configs = train_configs

        print(f'Loading data modules... (total: {len(self.train_configs)})')
        self.data_modules = [CartesianDataModule(train_configs=list(c), *args, **kwargs) for c in train_configs]
        self.total_steps = len(self.train_configs)
        super().__init__()

    @property
    def config(self):
        return self.data_modules[self.step].config

    @property
    def validation_datasets(self):
        return self.data_modules[self.step].validation_datasets

    @property
    def validation_dataset_mapping(self):
        return self.data_modules[self.step].validation_dataset_mapping

    @property
    def current_id(self):
        return self.train_configs[self.step][0]['id']

    def train_dataloader(self):
        return self.data_modules[self.step].train_dataloader()

    def val_dataloader(self):
        return self.data_modules[self.step].val_dataloader()


def _load_config(config):
    config = deepcopy(config)
    c_type = config['type']
    if c_type in {'fits', 'cartesian', 'sharp'}:
        data_path = config.pop('data_path')
        fits_paths, error_paths = _load_paths(data_path)
        ids = [os.path.basename(fp['Br']).split('.')[-3] for fp in fits_paths]
        dataset_type = 'sharp' if c_type == 'sharp' else 'fits'
        if error_paths is None:
            configs = [{**config, 'type': dataset_type, 'id': id, 'fits_path': fp}
                       for id, fp in zip(ids, fits_paths)]
        else:
            configs = [{**config, 'type': dataset_type, 'id': id, 'fits_path': fp, 'error_path': ep}
                       for id, fp, ep in zip(ids, fits_paths, error_paths)]
    elif c_type == 'muram_slice':
        data_path = config.pop('data_path')
        slices = sorted(glob.glob(data_path))
        ids = [os.path.basename(s).split('.')[-1] for s in slices]
        configs = [{**config, 'id': id, 'data_path': s} for id, s in zip(ids, slices)]
    else:
        raise NotImplementedError(f'Unknown data loader {c_type}')
    return configs


def _load_paths(data_path):
    if isinstance(data_path, list):
        results = [_load_paths(d) for d in data_path]
        fits_paths = [f for r in results for f in r[0]]
        error_paths = [f for r in results for f in r[1]] if all([r[1] is not None for r in results]) else None
    elif isinstance(data_path, str):
        p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
        t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
        r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
        err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
        err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
        err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

        assert len(p_files) == len(t_files) == len(r_files), f'Number of files in data path {data_path} does not match'
        fits_paths = list(zip(p_files, t_files, r_files))
        fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_paths]

        if len(err_p_files) > 0 or len(err_t_files) > 0 or len(err_r_files) > 0:
            assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(
                err_r_files), \
                f'Number of files in data path {data_path} does not match'
            error_paths = list(zip(err_p_files, err_t_files, err_r_files))
            error_paths = [{'Bp_err': d[0], 'Bt_err': d[1], 'Br_err': d[2]} for d in error_paths]
        else:
            error_paths = None
    elif isinstance(data_path, dict):
        p_files = sorted(glob.glob(data_path['Bp']))  # x
        t_files = sorted(glob.glob(data_path['Bt']))  # y
        r_files = sorted(glob.glob(data_path['Br']))  # z
        err_p_files = sorted(glob.glob(data_path['Bp_err'])) if 'Bp_err' in data_path else None  # x
        err_t_files = sorted(glob.glob(data_path['Bt_err'])) if 'Bt_err' in data_path else None  # y
        err_r_files = sorted(glob.glob(data_path['Br_err'])) if 'Br_err' in data_path else None  # z

        if err_p_files is not None and err_t_files is not None and err_r_files is not None:
            assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(
                err_r_files), \
                f'Number of files in data path {data_path} does not match'
            fits_paths = list(zip(p_files, t_files, r_files))
            fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2], } for d in fits_paths]
            error_paths = list(zip(err_p_files, err_t_files, err_r_files))
            error_paths = [{'Bp_err': d[0], 'Bt_err': d[1], 'Br_err': d[2], } for d in error_paths]
        else:
            assert len(p_files) == len(t_files) == len(r_files), \
                f'Number of files in data path {data_path} does not match'
            fits_paths = list(zip(p_files, t_files, r_files))
            fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_paths]
            error_paths = None
    else:
        raise NotImplementedError(f'Unknown data path type {type(data_path)}')
    return fits_paths, error_paths


class SHARPDataset(FITSDataset):

    def __init__(self, **kwargs):
        super().__init__(Mm_per_pixel=.36, **kwargs)


class NumpyDataset(MapDataset):

    def __init__(self, data_path, **kwargs):
        data = np.load(data_path)
        bx = data['bx']
        by = data['by']
        bz = data['bz']
        bx_err = data.get('bx_err', None)
        by_err = data.get('by_err', None)
        bz_err = data.get('bz_err', None)
        b = np.stack([bx, by, bz], axis=-1)
        if bx_err is not None and by_err is not None and bz_err is not None:
            b_err = np.stack([bx_err, by_err, bz_err], axis=-1)
        else:
            b_err = None

        super().__init__(b=b, b_err=b_err, **kwargs)


class PotentialBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, coord_range, ds_per_pixel, G_per_dB, strides=1, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides)
        coords = coords * ds_per_pixel
        b_err = b_err / G_per_dB
        b = b / G_per_dB

        # adjust coordinates in xy plane
        c_x_min, c_x_max = coords[..., 0].min(), coords[..., 0].max()
        c_y_min, c_y_max = coords[..., 1].min(), coords[..., 1].max()
        target_x_min, target_x_max = coord_range[0]
        target_y_min, target_y_max = coord_range[1]
        coords[..., 0] = (coords[..., 0] - c_x_min) / (c_x_max - c_x_min) * (target_x_max - target_x_min) + target_x_min
        coords[..., 1] = (coords[..., 1] - c_y_min) / (c_y_max - c_y_min) * (target_y_max - target_y_min) + target_y_min

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)


class PotentialTopBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, coord_range, ds_per_pixel, G_per_dB, strides=2, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides,
                                                         only_top=True, progress=False)
        coords = coords * ds_per_pixel
        b_err = b_err / G_per_dB
        b = b / G_per_dB

        # adjust coordinates in xy plane
        c_x_min, c_x_max = coords[..., 0].min(), coords[..., 0].max()
        c_y_min, c_y_max = coords[..., 1].min(), coords[..., 1].max()
        target_x_min, target_x_max = coord_range[0]
        target_y_min, target_y_max = coord_range[1]
        coords[..., 0] = (coords[..., 0] - c_x_min) / (c_x_max - c_x_min) * (target_x_max - target_x_min) + target_x_min
        coords[..., 1] = (coords[..., 1] - c_y_min) / (c_y_max - c_y_min) * (target_y_max - target_y_min) + target_y_min

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)


def process_map(map, slice, bin):
    if slice:
        map = map.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                         top_right=u.Quantity((slice[1], slice[3]), u.pixel))
    if bin > 1:
        map = map.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean)
    return map
