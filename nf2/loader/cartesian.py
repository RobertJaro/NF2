import copy
import gc
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from astropy import units as u
from lightning.pytorch import LightningDataModule

from nf2.data.dataset import CubeDataset, RandomCoordinateDataset, RandomHeightCoordinateDataset, SlicesDataset
from nf2.loader.base import BaseDataModule, DEFAULT_NUM_WORKERS
from nf2.loader.muram import MURaMDataset, MURaMCubeDataset
from nf2.loader.cartesian_datasets import (
    AnalyticalBoundaryDataset, FITSDataset, FldIncAziFITSDataset, LosFITSDataset,
    LosTrvAziFITSDataset, NumpyDataset, PotentialBoundaryDataset, PotentialTopBoundaryDataset,
    SHARPDataset,
)


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
        num_workers = num_workers if num_workers is not None else DEFAULT_NUM_WORKERS
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


def _load_cartesian_data_module(worker_args):
    step, total_steps, boundaries, args, kwargs = worker_args
    print(f'Loading data module {step + 1:03d}/{total_steps:03d}; '
          f'ID: {CartesianSeriesDataModule._step_id(boundaries, step)}', flush=True)
    return CartesianDataModule(boundaries=list(boundaries), *args, **kwargs)


def _without_overview_plots(boundaries):
    boundaries = deepcopy(boundaries)
    for config in boundaries:
        config.setdefault('plot', False)
    return boundaries


def _series_work_path(work_path, step, include_rank=False):
    parts = [work_path, 'series_data_modules', f'{step:06d}']
    if include_rank:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK'))
        if world_size > 1 and rank is not None:
            parts.append(f'rank_{int(rank):03d}')
    return os.path.join(*parts)


class CartesianSeriesDataModule(LightningDataModule):

    def __init__(self, boundaries, current_step, data_module_workers=None, preload_data_modules=True, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.preload_data_modules = preload_data_modules
        expanded_boundaries = [_load_config(c) for c in boundaries]
        assert all([len(c) == len(expanded_boundaries[0]) for c in expanded_boundaries]), \
            'Inconsistent number of training files in configurations. Check your configurations.'
        boundaries = list(zip(*expanded_boundaries))

        # remove already extrapolated files
        assert current_step < len(boundaries), \
            'Not enough data files found to continue training. Training completed or configuration is incorrect.'

        self.step = current_step
        self.boundaries = boundaries
        self.total_steps = len(self.boundaries)
        self.data_modules = [None] * self.total_steps
        super().__init__()
        self._load_data_modules(data_module_workers)

    def _kwargs_for_step(self, step):
        kwargs = deepcopy(self.kwargs)
        if kwargs.get('work_path') is not None:
            kwargs['work_path'] = _series_work_path(kwargs['work_path'], step, include_rank=not self.preload_data_modules)
        return kwargs

    @staticmethod
    def _step_id(boundaries, step):
        for config in boundaries:
            if 'step_id' in config and config['step_id'] is not None:
                return config['step_id']
            if 'id' in config and config['id'] is not None:
                return config['id']
        return f'step_{step:06d}'

    def _load_data_modules(self, data_module_workers):
        if not self.preload_data_modules:
            print(f'Loading data modules lazily... (total: {len(self.boundaries)})')
            self._get_data_module(self.step)
            return

        n_workers = data_module_workers if data_module_workers is not None else self.kwargs.get(
            'num_workers', DEFAULT_NUM_WORKERS)
        n_workers = max(1, min(n_workers, self.total_steps))
        print(f'Loading data modules... (total: {len(self.boundaries)}, workers: {n_workers})')
        worker_args = [(step, self.total_steps,
                        boundaries if step == self.step else _without_overview_plots(boundaries),
                        self.args, self._kwargs_for_step(step))
                       for step, boundaries in enumerate(self.boundaries)]
        if n_workers == 1:
            for args in worker_args:
                step = args[0]
                self.data_modules[step] = _load_cartesian_data_module(args)
            return

        n_workers = max(1, min(n_workers, len(worker_args)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for args, data_module in zip(worker_args, pool.map(_load_cartesian_data_module, worker_args)):
                step = args[0]
                self.data_modules[step] = data_module

    def _get_data_module(self, step):
        if self.data_modules[step] is None:
            self._evict_data_modules(keep_step=step)
            self.data_modules[step] = _load_cartesian_data_module(
                (step, self.total_steps, self.boundaries[step], self.args, self._kwargs_for_step(step)))
        return self.data_modules[step]

    def _evict_data_modules(self, keep_step):
        for i, data_module in enumerate(self.data_modules):
            if i == keep_step or data_module is None:
                continue
            data_module.clear()
            self.data_modules[i] = None
        gc.collect()

    @property
    def config(self):
        return self._get_data_module(self.step).config

    @property
    def validation_datasets(self):
        return self._get_data_module(self.step).validation_datasets

    @property
    def validation_dataset_mapping(self):
        return self._get_data_module(self.step).validation_dataset_mapping

    @property
    def current_id(self):
        return self._step_id(self.boundaries[self.step], self.step)

    def activate_step(self, step):
        self.step = step
        if not self.preload_data_modules:
            self.data_modules = [None] * self.total_steps
        self._get_data_module(self.step)

    def train_dataloader(self):
        return self._get_data_module(self.step).train_dataloader()

    def val_dataloader(self):
        return self._get_data_module(self.step).val_dataloader()


def _load_config(config):
    config = deepcopy(config)
    c_type = config['type']
    if c_type in {'fits', 'sharp', 'los_trv_azi', 'los', 'fld_inc_azi'}:
        path_config = config.pop('fits_path', config.pop('data_path', None))
        error_config = config.pop('error_path', None)
        if path_config is None:
            raise ValueError(f"Series dataset '{config.get('id', c_type)}' requires files or data_path.")
        fits_paths, error_paths = _load_paths(path_config, c_type)
        if error_config is not None:
            error_paths = _load_error_paths(error_config, c_type, len(fits_paths))
        if error_paths is None:
            error_paths = [{} for _ in fits_paths]
        ids = [_series_id(fp) for fp in fits_paths]
        configs = [
            {
                **config,
                'id': config.get('id', id),
                'step_id': id,
                'fits_path': fp,
                **({'error_path': ep} if ep else {}),
            }
            for id, fp, ep in zip(ids, fits_paths, error_paths)
        ]
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


def _load_error_paths(error_path, data_type, expected_length=None):
    error_keys = [] if data_type in {'los_trv_azi', 'los', 'fld_inc_azi'} else ['Bp_err', 'Bt_err', 'Br_err']
    if not error_keys:
        return None
    if isinstance(error_path, list):
        results = [_load_error_paths(path, data_type) for path in error_path]
        error_paths = [path for paths in results for path in paths] if all(r is not None for r in results) else None
        if expected_length is not None and error_paths is not None and len(error_paths) != expected_length:
            raise ValueError(f'Inconsistent number of error files in {error_path}: '
                             f'expected {expected_length}, got {len(error_paths)}.')
        return error_paths
    if isinstance(error_path, str):
        error_files_by_key = {
            key: sorted(glob.glob(os.path.join(error_path, f'*{key}.fits')))
            for key in error_keys
        }
    elif isinstance(error_path, dict):
        error_files_by_key = {key: _expand_component_files(error_path[key]) for key in error_keys}
    else:
        raise NotImplementedError(f'Unknown error path type {type(error_path)}')
    _assert_component_lengths(error_files_by_key, error_path, expected_length=expected_length)
    return [{key: error_files_by_key[key][i] for key in error_keys} for i in range(expected_length)]


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
        files_by_key = {key: _expand_component_files(data_path[key]) for key in component_keys}
        n_files = _assert_component_lengths(files_by_key, data_path)
        fits_paths = [{key: files_by_key[key][i] for key in component_keys} for i in range(n_files)]

        if error_keys and all(key in data_path for key in error_keys):
            error_files_by_key = {key: _expand_component_files(data_path[key]) for key in error_keys}
            _assert_component_lengths(error_files_by_key, data_path, expected_length=n_files)
            error_paths = [{key: error_files_by_key[key][i] for key in error_keys} for i in range(n_files)]
        else:
            error_paths = None
    else:
        raise NotImplementedError(f'Unknown data path type {type(data_path)}')
    return fits_paths, error_paths


def _expand_component_files(value):
    if isinstance(value, list):
        files = []
        for item in value:
            files.extend(_expand_component_files(item))
        return files
    if isinstance(value, str):
        return sorted(glob.glob(value))
    raise NotImplementedError(f'Unknown component path type {type(value)}')


def _assert_component_lengths(files_by_key, source, expected_length=None):
    lengths = {key: len(files) for key, files in files_by_key.items()}
    if expected_length is None:
        expected_length = next(iter(lengths.values()))
    if any(length != expected_length for length in lengths.values()):
        raise AssertionError(f'Number of files in data path {source} does not match: {lengths}')
    if expected_length == 0:
        raise AssertionError(f'No files found in data path {source}')
    return expected_length
