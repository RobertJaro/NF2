import copy
import glob
import os
from copy import deepcopy

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from lightning.pytorch import LightningDataModule
from sunpy.map import Map

from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, RandomHeightCoordinateDataset, SlicesDataset, \
    TensorsDataset
from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.loader import load_potential_field_boundary
from nf2.loader.base import BaseDataModule, MapDataset
from nf2.loader.muram import MURaMDataset, MURaMCubeDataset


def _combined_xy_range(datasets):
    ranges = [dataset.coord_range for dataset in datasets if hasattr(dataset, 'coord_range')]
    if not ranges:
        raise ValueError('At least one Cartesian boundary dataset must provide coord_range.')
    return np.array([
        [min(r[0, 0] for r in ranges), max(r[0, 1] for r in ranges)],
        [min(r[1, 0] for r in ranges), max(r[1, 1] for r in ranges)],
    ], dtype=np.float32)


def _potential_coord_range(bottom_boundary_dataset, domain_coord_range):
    coord_range = np.array(domain_coord_range, dtype=np.float32, copy=True)
    coord_range[:2] = bottom_boundary_dataset.coord_range
    return coord_range


class CartesianDataModule(BaseDataModule):

    def __init__(self, boundaries, work_path, validation=None,
                 potential_boundary=None, sampler=None,
                 Mm_per_ds=100, Gauss_per_dB=1000, z_range=None, validation_batch_size=2 ** 14, log_shape=False,
                 batch_size=2 ** 13, validation_pixel_per_ds=128, iterations=None,
                 num_workers=None, type=None, geometry=None, **kwargs):
        self.Mm_per_ds = Mm_per_ds
        self.Gauss_per_dB = Gauss_per_dB
        self.work_path = work_path
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_pixel_per_ds = validation_pixel_per_ds
        # wrap data if only one slice is provided
        boundaries = boundaries if isinstance(boundaries, list) else [boundaries]
        self.ds_batch_size = batch_size // len(boundaries)
        # boundary dataset
        num_workers = num_workers if num_workers is not None else os.cpu_count()
        training_datasets = [self._load_ds_config(config) for config in boundaries]
        training_datasets = dict(training_datasets)  # convert to dict

        bottom_boundary_dataset = list(training_datasets.values())[0]

        # random sampling dataset
        coord_range = _combined_xy_range(training_datasets.values())
        z_range = [0, 100] if z_range is None else z_range
        z_range_arr = np.array([z_range]) / Mm_per_ds
        coord_range = np.concatenate([coord_range, z_range_arr], axis=0)
        sampler = copy.deepcopy(sampler) if sampler is not None else {'batch_size': 2 ** 14}
        sampler_state = deepcopy(sampler)
        random_type = sampler.pop('type', 'default')
        random_requires_jacobian = sampler.pop('requires_jacobian', True)
        if random_type == 'default':
            random_dataset = RandomCoordinateDataset(coord_range, length=iterations,
                                                     requires_jacobian=random_requires_jacobian, **sampler)
        elif random_type == 'height':
            random_dataset = RandomHeightCoordinateDataset(coord_range, length=iterations,
                                                           requires_jacobian=random_requires_jacobian, **sampler)
        else:
            raise ValueError(f'Unknown random dataset type: {random_type}')
        random_dataset.config = {
            'id': 'random',
            'type': random_type,
            'role': 'training',
            'requires_jacobian': random_requires_jacobian,
            'coord_range': coord_range,
            **sampler_state,
        }

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
        potential_boundary = potential_boundary if potential_boundary is not None else \
            {'type': 'potential', 'strides': 4}
        potential_boundary = deepcopy(potential_boundary)
        potential_boundary_state = deepcopy(potential_boundary)
        boundary_batch_size = potential_boundary.pop('batch_size', batch_size // 4)
        boundary_requires_jacobian = potential_boundary.pop('requires_jacobian', False)
        boundary_type = potential_boundary.pop('type')
        if boundary_type == 'none':
            pass
        elif boundary_type == 'potential':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            potential_coord_range = _potential_coord_range(bottom_boundary_dataset, coord_range)
            boundary_ds = PotentialBoundaryDataset(bz=bz,
                                                   height_pixel=coord_range[2, -1] / ds_per_pixel,
                                                   coord_range=potential_coord_range,
                                                   ds_per_pixel=ds_per_pixel, Gauss_per_dB=Gauss_per_dB,
                                                   work_path=work_path,
                                                   requires_jacobian=boundary_requires_jacobian,
                                                   batch_size=boundary_batch_size, **potential_boundary)
            boundary_ds.config = {
                'id': 'potential',
                'type': boundary_type,
                'role': 'training',
                'requires_jacobian': boundary_requires_jacobian,
                'batch_size': boundary_batch_size,
                'coord_range': potential_coord_range,
                **potential_boundary_state,
            }
            training_datasets['potential'] = boundary_ds
        elif boundary_type == 'potential_top':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            potential_coord_range = _potential_coord_range(bottom_boundary_dataset, coord_range)
            boundary_ds = PotentialTopBoundaryDataset(bz=bz,
                                                      height_pixel=coord_range[2, -1] / ds_per_pixel,
                                                      coord_range=potential_coord_range,
                                                      ds_per_pixel=ds_per_pixel, Gauss_per_dB=Gauss_per_dB,
                                                      work_path=work_path,
                                                      requires_jacobian=boundary_requires_jacobian,
                                                      batch_size=boundary_batch_size, **potential_boundary)
            boundary_ds.config = {
                'id': 'potential',
                'type': boundary_type,
                'role': 'training',
                'requires_jacobian': boundary_requires_jacobian,
                'batch_size': boundary_batch_size,
                'coord_range': potential_coord_range,
                **potential_boundary_state,
            }
            training_datasets['potential'] = boundary_ds
        else:
            raise ValueError(f'Unknown boundary type: {potential_boundary["type"]}')

        # validation datasets
        validation_datasets = {}
        if validation is None:
            validation = boundaries.copy()
            validation.append({'type': "cube", 'id': 'cube', })
            validation.append({'type': "slices", 'id': 'slices', })

        validation_datasets = [self._load_valid_ds_config(config) for config in validation]
        validation_datasets = dict(validation_datasets)  # convert to dict

        config = {'schema_version': '0.4',
                  'type': 'cartesian',
                  'geometry': 'cartesian',
                  'coordinate_system': 'cartesian',
                  'field_components': ['Bx', 'By', 'Bz'],
                  'field_unit': 'G',
                  'length_unit': 'Mm',
                  'max_height': z_range[1],
                  'z_range': z_range,
                  'z_range_ds': coord_range[2],
                  'Mm_per_ds': Mm_per_ds, 'Gauss_per_dB': Gauss_per_dB,
                  'normalization': {'Mm_per_ds': Mm_per_ds, 'Gauss_per_dB': Gauss_per_dB},
                  'coord_range': [], 'ds_per_pixel': [], 'height_mapping': [], 'wcs': [],
                  'cube_shape': []}
        for dataset in training_datasets.values():
            if not all(hasattr(dataset, attr) for attr in ['coord_range', 'cube_shape', 'ds_per_pixel']):
                continue
            config['coord_range'].append(dataset.coord_range)
            config['cube_shape'].append(dataset.cube_shape)
            config['ds_per_pixel'].append(dataset.ds_per_pixel)
            config['height_mapping'].append(getattr(dataset, 'height_mapping', None))
            config['wcs'].append(getattr(dataset, 'wcs', None))

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
        elif type == 'analytical':
            return AnalyticalBoundaryDataset(**kwargs)
        elif type == 'muram_slice':
            return MURaMDataset(**kwargs)
        elif type == 'muram_cube':
            return MURaMCubeDataset(**kwargs)
        elif type == 'slices':
            return SlicesDataset(**kwargs)
        elif type == 'cube':
            return CubeDataset(**kwargs)
        else:
            raise ValueError(f'Unknown boundary type: {type}. Supported types: '
                             f'fits, los_trv_azi, los, sharp, fld_inc_azi, numpy, muram_slice, muram_cube')

    def _load_ds_config(self, config):
        """
        Load dataset from configuration.
        :param config: dataset configuration
        :return: dataset instance
        """
        config = deepcopy(config)
        ds_type = config['type']
        dataset_id = config.pop('id', f'train_{ds_type}')
        requires_jacobian = config.pop('requires_jacobian', False)
        config['batch_size'] = config.pop('batch_size', self.ds_batch_size)  # default batch size
        dataset = self.init_dataset(**config, Mm_per_ds=self.Mm_per_ds, Gauss_per_dB=self.Gauss_per_dB,
                                    work_path=self.work_path, requires_jacobian=requires_jacobian)
        dataset.config = {
            'id': dataset_id,
            'type': ds_type,
            'role': 'training',
            'requires_jacobian': requires_jacobian,
            **deepcopy(config),
        }
        return dataset_id, dataset

    def _load_valid_ds_config(self, config):
        """
        Load validation dataset from configuration.
        :param config: dataset configuration
        :return: dataset instance
        """
        config = deepcopy(config)
        ds_type = config['type']
        dataset_id = config.pop('id', f'valid_{ds_type}')
        config['batch_size'] = config.pop('batch_size', self.validation_batch_size)
        plot = config.pop('plot', False)
        dataset = self.init_dataset(**config,
                                    Mm_per_ds=self.Mm_per_ds, Gauss_per_dB=self.Gauss_per_dB,
                                    shuffle=False, filter_nans=False,
                                    work_path=self.work_path, plot=plot,
                                    ds_per_pixel=1 / self.validation_pixel_per_ds, coord_range=self.coord_range)
        dataset.config = {
            'id': dataset_id,
            'type': ds_type,
            'role': 'validation',
            'plot': plot,
            **deepcopy(config),
        }
        return dataset_id, dataset


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

    def __init__(self, boundaries, current_step, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        expanded_boundaries = [_load_config(c) for c in boundaries]
        assert all([len(c) == len(expanded_boundaries[0]) for c in expanded_boundaries]), \
            'Inconsistent number of training files in configurations. Check your configurations.'
        boundaries = list(zip(*expanded_boundaries))

        # remove already extrapolated files
        assert current_step < len(boundaries), \
            'Not enough data files found to continue training. Training completed or configuration is incorrect.'

        self.step = current_step
        self.boundaries = boundaries

        print(f'Loading data modules... (total: {len(self.boundaries)})')
        self.data_modules = [CartesianDataModule(boundaries=list(c), *args, **kwargs) for c in boundaries]
        self.total_steps = len(self.boundaries)
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
        return self.boundaries[self.step][0].get('step_id', self.boundaries[self.step][0]['id'])

    def train_dataloader(self):
        return self.data_modules[self.step].train_dataloader()

    def val_dataloader(self):
        return self.data_modules[self.step].val_dataloader()


def _load_config(config):
    config = deepcopy(config)
    c_type = config['type']
    if c_type in {'fits', 'sharp', 'los_trv_azi', 'los', 'fld_inc_azi'}:
        path_config = config.pop('fits_path', config.pop('data_path', None))
        if path_config is None:
            raise ValueError(f"Series dataset '{config.get('id', c_type)}' requires files or data_path.")
        fits_paths, error_paths = _load_paths(path_config, c_type)
        if error_paths is None:
            error_paths = [{} for _ in fits_paths]
        ids = [_series_id(fp) for fp in fits_paths]
        configs = [{**config, 'id': config.get('id', id), 'step_id': id, 'fits_path': {**fp, **ep}}
                   for id, fp, ep in zip(ids, fits_paths, error_paths)]
    elif c_type == 'muram_slice':
        data_path = config.pop('data_path')
        slices = sorted(glob.glob(data_path))
        ids = [os.path.basename(s).split('.')[-1] for s in slices]
        configs = [{**config, 'id': id, 'data_path': s} for id, s in zip(ids, slices)]
    else:
        raise NotImplementedError(f'Unknown data loader {c_type}')
    return configs


def _series_id(files):
    file_path = files.get('Br') or files.get('B_los') or next(iter(files.values()))
    stem = os.path.basename(file_path)
    if stem.endswith('.fits'):
        stem = stem[:-5]
    for suffix in ['.Br', '.Bt', '.Bp', '.B_los', '.B_trv', '.B_azi', '_Br', '_Bt', '_Bp',
                   '_B_los', '_B_trv', '_B_azi']:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _load_paths(data_path, data_type='fits'):
    if data_type == 'los_trv_azi':
        return _load_component_paths(data_path, ['B_los', 'B_trv', 'B_azi'], [])
    if data_type == 'los':
        return _load_component_paths(data_path, ['B_los'], [])
    if data_type == 'fld_inc_azi':
        return _load_component_paths(data_path, ['B_fld', 'B_inc', 'B_azi'], [])
    return _load_component_paths(data_path, ['Bp', 'Bt', 'Br'], ['Bp_err', 'Bt_err', 'Br_err'])


def _load_component_paths(data_path, component_keys, error_keys):
    if isinstance(data_path, list):
        results = [_load_component_paths(d, component_keys, error_keys) for d in data_path]
        fits_paths = [f for r in results for f in r[0]]
        error_paths = [f for r in results for f in r[1]] if all([r[1] is not None for r in results]) else None
    elif isinstance(data_path, str):
        files_by_key = {
            key: sorted(glob.glob(os.path.join(data_path, f'*{key}.fits')))
            for key in component_keys
        }
        n_files = _assert_component_lengths(files_by_key, data_path)
        fits_paths = [{key: files_by_key[key][i] for key in component_keys} for i in range(n_files)]

        error_files_by_key = {
            key: sorted(glob.glob(os.path.join(data_path, f'*{key}.fits')))
            for key in error_keys
        }
        if error_keys and any(len(files) > 0 for files in error_files_by_key.values()):
            _assert_component_lengths(error_files_by_key, data_path, expected_length=n_files)
            error_paths = [{key: error_files_by_key[key][i] for key in error_keys} for i in range(n_files)]
        else:
            error_paths = None
    elif isinstance(data_path, dict):
        files_by_key = {key: sorted(glob.glob(data_path[key])) for key in component_keys}
        n_files = _assert_component_lengths(files_by_key, data_path)
        fits_paths = [{key: files_by_key[key][i] for key in component_keys} for i in range(n_files)]

        if error_keys and all(key in data_path for key in error_keys):
            error_files_by_key = {key: sorted(glob.glob(data_path[key])) for key in error_keys}
            _assert_component_lengths(error_files_by_key, data_path, expected_length=n_files)
            error_paths = [{key: error_files_by_key[key][i] for key in error_keys} for i in range(n_files)]
        else:
            error_paths = None
    else:
        raise NotImplementedError(f'Unknown data path type {type(data_path)}')
    return fits_paths, error_paths


def _assert_component_lengths(files_by_key, source, expected_length=None):
    lengths = {key: len(files) for key, files in files_by_key.items()}
    if expected_length is None:
        expected_length = next(iter(lengths.values()))
    if any(length != expected_length for length in lengths.values()):
        raise AssertionError(f'Number of files in data path {source} does not match: {lengths}')
    if expected_length == 0:
        raise AssertionError(f'No files found in data path {source}')
    return expected_length


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


class AnalyticalBoundaryDataset(TensorsDataset):

    def __init__(self, case, boundary='bottom', resolution=64, bounds=None,
                 batch_size=2 ** 13, Gauss_per_dB=1000, Mm_per_ds=100,
                 work_path=None, **kwargs):
        bounds = [-1, 1, -1, 1, 0, 2] if bounds is None else bounds
        if case == 1:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi / 4,
                                          resolution=resolution, bounds=bounds)
        elif case == 2:
            res = resolution if isinstance(resolution, list) else [80, 80, 72]
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15,
                                          resolution=res, bounds=bounds)
        else:
            raise ValueError(f'Invalid analytical case {case}. Available cases: 1, 2')

        coord_cube = np.stack(np.meshgrid(
            np.linspace(bounds[0], bounds[1], b_cube.shape[0], dtype=np.float32),
            np.linspace(bounds[2], bounds[3], b_cube.shape[1], dtype=np.float32),
            np.linspace(bounds[4], bounds[5], b_cube.shape[2], dtype=np.float32),
            indexing='ij'), -1)

        if boundary == 'bottom':
            coords = coord_cube[:, :, 0].reshape((-1, 3))
            values = b_cube[:, :, 0].reshape((-1, 3))
        elif boundary == 'full':
            coords = np.concatenate([
                coord_cube[:, :, 0].reshape((-1, 3)),
                coord_cube[:, :, -1].reshape((-1, 3)),
                coord_cube[:, 0, :].reshape((-1, 3)),
                coord_cube[:, -1, :].reshape((-1, 3)),
                coord_cube[0, :, :].reshape((-1, 3)),
                coord_cube[-1, :, :].reshape((-1, 3)),
            ])
            values = np.concatenate([
                b_cube[:, :, 0].reshape((-1, 3)),
                b_cube[:, :, -1].reshape((-1, 3)),
                b_cube[:, 0, :].reshape((-1, 3)),
                b_cube[:, -1, :].reshape((-1, 3)),
                b_cube[0, :, :].reshape((-1, 3)),
                b_cube[-1, :, :].reshape((-1, 3)),
            ])
        else:
            raise ValueError("Analytical boundary must be 'bottom' or 'full'.")

        b_scale = np.nanmax(np.abs(b_cube))
        b_scale = 1 if b_scale == 0 else b_scale
        values = values / b_scale

        self.bz = b_cube[:, :, 0, 2] / b_scale * Gauss_per_dB
        self.ds_per_pixel = (bounds[1] - bounds[0]) / max(b_cube.shape[0] - 1, 1)
        self.coord_range = np.array([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], dtype=np.float32)
        self.ds = np.array([self.ds_per_pixel, self.ds_per_pixel], dtype=np.float32)
        self.cube_shape = b_cube.shape[:2]
        self.los_trv_azi = False
        self.height_mapping = None
        self.wcs = None

        super().__init__({'coords': coords, 'b_true': values.astype(np.float32)},
                         batch_size=batch_size, work_path=work_path,
                         ds_name=f'analytical_case_{case}', **kwargs)


class PotentialBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, coord_range, ds_per_pixel, Gauss_per_dB, strides=1, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides)
        coords = coords * ds_per_pixel
        b_err = b_err / Gauss_per_dB
        b = b / Gauss_per_dB

        # adjust coordinates in xy plane
        c_x_min, c_x_max = coords[..., 0].min(), coords[..., 0].max()
        c_y_min, c_y_max = coords[..., 1].min(), coords[..., 1].max()
        target_x_min, target_x_max = coord_range[0]
        target_y_min, target_y_max = coord_range[1]
        coords[..., 0] = (coords[..., 0] - c_x_min) / (c_x_max - c_x_min) * (target_x_max - target_x_min) + target_x_min
        coords[..., 1] = (coords[..., 1] - c_y_min) / (c_y_max - c_y_min) * (target_y_max - target_y_min) + target_y_min

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)


class PotentialTopBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, coord_range, ds_per_pixel, Gauss_per_dB, strides=2, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides,
                                                         only_top=True, progress=False)
        coords = coords * ds_per_pixel
        b_err = b_err / Gauss_per_dB
        b = b / Gauss_per_dB

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
