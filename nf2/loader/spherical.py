import gc
import glob
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from astropy import units as u
from lightning.pytorch import LightningDataModule

from nf2.data.dataset import RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset, RandomRadialGroupedCoordinateDataset
from nf2.loader.base import BaseDataModule, DEFAULT_NUM_WORKERS
from nf2.loader.spherical_datasets import (
    SphericalFITSReferenceDataset, SphericalMapDataset, _map, _reference_map_coordinate_bounds,
    spherical_coord_scale,
)


class SphericalDataModule(BaseDataModule):

    def __init__(self, boundaries, validation, samplers=None,
                 max_radius=1.3,
                 Mm_per_ds=100,
                 Gauss_per_dB=1000, work_path=None,
                 batch_size=4096, type=None, geometry=None, overview_id=None, **kwargs):

        self.ds_mapping = {'map': SphericalMapDataset,
                           'fits_reference': SphericalFITSReferenceDataset,
                           'random_spherical': RandomSphericalCoordinateDataset,
                           'random_radial_grouped': RandomRadialGroupedCoordinateDataset,
                           'sphere': SphereDataset,
                           'spherical_slices': SphereSlicesDataset}

        # data parameters
        self.Gauss_per_dB = Gauss_per_dB
        self.Mm_per_ds = Mm_per_ds
        self.cube_shape = [1, max_radius]
        self.spatial_norm = 1 * u.solRad
        self.overview_id = overview_id

        # init boundary datasets
        general_config = {'work_path': work_path, 'batch_size': batch_size, 'Gauss_per_dB': Gauss_per_dB,
                          'radius_range': [1, max_radius], 'Mm_per_ds': Mm_per_ds}

        config = {'schema_version': '0.4',
                  'type': 'spherical',
                  'geometry': 'spherical',
                  'coordinate_system': 'heliographic_carrington',
                  'boundary_field_components': ['Br', 'Btheta', 'Bphi'],
                  'model_field_components': ['Bx', 'By', 'Bz'],
                  'field_unit': 'G',
                  'length_unit': 'Mm',
                  'radius_unit': 'solRad',
                  'max_radius': max_radius,
                  'radius_range': [1, max_radius],
                  'spatial_norm_Mm': (1 * u.solRad).to_value(u.Mm),
                  'Gauss_per_dB': Gauss_per_dB,
                  'Mm_per_ds': Mm_per_ds,
                  'normalization': {'Mm_per_ds': Mm_per_ds, 'Gauss_per_dB': Gauss_per_dB}}

        training_configs = list(boundaries) + list(samplers or [])
        training_datasets = self.load_config(training_configs, general_config, prefix='train')
        validation_datasets = self.load_config(validation, general_config, prefix='validation')
        self._validate_coordinate_scaling({**training_datasets, **validation_datasets}, Mm_per_ds)

        super().__init__(training_datasets, validation_datasets, config, **kwargs)

    def load_config(self, configs, general_config, prefix='train'):
        datasets = OrderedDict()
        for i, config in enumerate(configs):
            config = deepcopy(config)
            c_type = config.pop('type')
            c_name = config.pop('id') if 'id' in config else f'{prefix}_{c_type}_{i}'
            requires_jacobian = config.pop('requires_jacobian', True)
            config['ds_name'] = c_name if self.overview_id is None else f'{self.overview_id}_{c_name}'
            # update config with general config
            for k, v in general_config.items():
                if k not in config:
                    config[k] = v
            if c_type in ['random_spherical', 'random_radial_grouped']:
                self._apply_random_reference_map(config)
            config['requires_jacobian'] = requires_jacobian
            os.makedirs(config['work_path'], exist_ok=True)
            dataset = self.ds_mapping[c_type](**config)
            dataset.config = {
                'id': c_name,
                'type': c_type,
                'role': prefix,
                **deepcopy(config),
            }
            datasets[c_name] = dataset
        return datasets

    @staticmethod
    def _validate_coordinate_scaling(datasets, Mm_per_ds):
        expected_scale = spherical_coord_scale(Mm_per_ds)
        for name, dataset in datasets.items():
            coord_scale = getattr(dataset, 'coord_scale', expected_scale)
            if abs(coord_scale - expected_scale) > max(1e-6, abs(expected_scale) * 1e-8):
                raise ValueError(
                    f"Spherical dataset '{name}' uses coordinate scale {coord_scale}, "
                    f"expected {expected_scale} from Mm_per_ds={Mm_per_ds}."
                )

    @staticmethod
    def _apply_random_reference_map(config):
        reference_config = config.pop('reference_map', None)
        if reference_config is None:
            return

        if isinstance(reference_config, str):
            reference_config = {'file': reference_config}
        if not isinstance(reference_config, dict):
            raise TypeError('reference_map must be a file path or configuration dictionary.')

        reference_file = reference_config.get('file')
        if reference_file is None:
            raise ValueError('reference_map requires a file.')

        reference_map = _map(reference_file)
        mu_filter = reference_config.get('mu_filter', reference_config.get('mu'))
        bounds = _reference_map_coordinate_bounds(reference_map, mu_filter, f'reference_map file: {reference_file}')

        config.setdefault('latitude_range', bounds['latitude_range'])
        config.setdefault('longitude_range', bounds['longitude_range'])
        config.setdefault('unit', 'deg')


def _load_spherical_data_module(worker_args):
    step, total_steps, boundaries, args, kwargs = worker_args
    print(f'Loading data module {step + 1:03d}/{total_steps:03d}; '
          f'ID: {SphericalSeriesDataModule._step_id(boundaries, step)}', flush=True)
    return SphericalDataModule(boundaries=boundaries, *args, **kwargs)


def _without_overview_plots(configs):
    configs = deepcopy(configs)
    for config in configs:
        config.setdefault('plot_overview', False)
        config.setdefault('plot', False)
    return configs


def _series_work_path(work_path, step, include_rank=False):
    parts = [work_path, 'series_data_modules', f'{step:06d}']
    if include_rank:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK'))
        if world_size > 1 and rank is not None:
            parts.append(f'rank_{int(rank):03d}')
    return os.path.join(*parts)


class SphericalSeriesDataModule(LightningDataModule):

    def __init__(self, boundaries, samplers=None, current_step=0, iterations=None, data_module_workers=None,
                 preload_data_modules=True, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.iterations = iterations
        self.preload_data_modules = preload_data_modules
        self.validation_configs = deepcopy(self.kwargs.pop('validation'))

        self.boundaries = self._expand_boundaries(list(boundaries) + list(samplers or []))
        self.validation = self._expand_validation(self.validation_configs, self.boundaries)
        self.step = current_step
        self.total_steps = len(self.boundaries)

        assert self.step < self.total_steps, \
            'Not enough data files found to continue training. Training completed or configuration is incorrect.'

        self.data_modules = [None] * self.total_steps
        super().__init__()
        self._load_data_modules(data_module_workers)

    def _expand_boundaries(self, boundaries):
        expanded_configs = [self._expand_config(config) for config in boundaries]
        total_steps = max(len(configs) for configs in expanded_configs)
        assert all(len(configs) in (1, total_steps) for configs in expanded_configs), \
            'Inconsistent number of training files in configurations. Check your configurations.'

        configs_by_step = []
        for step in range(total_steps):
            step_configs = [deepcopy(configs[step] if len(configs) > 1 else configs[0])
                            for configs in expanded_configs]
            context = {config.get('id', f'train_{i}'): config
                       for i, config in enumerate(step_configs)}
            configs_by_step.append([self._resolve_placeholders(config, context) for config in step_configs])
        return configs_by_step

    def _expand_validation(self, validation, boundaries_by_step):
        validation = list(validation)
        expanded_configs = [self._expand_config(config) for config in validation]
        total_steps = len(boundaries_by_step)
        assert all(len(configs) in (1, total_steps) for configs in expanded_configs), \
            'Inconsistent number of validation files in configurations. Check your configurations.'

        validation_by_step = []
        for step, step_boundaries in enumerate(boundaries_by_step):
            context = {config.get('id', f'train_{i}'): config
                       for i, config in enumerate(step_boundaries)}
            step_validation = []
            for configs in expanded_configs:
                config = deepcopy(configs[step] if len(configs) > 1 else configs[0])
                step_validation.append(self._resolve_placeholders(config, context))
            validation_by_step.append(step_validation)
        return validation_by_step

    def _kwargs_for_step(self, step):
        kwargs = deepcopy(self.kwargs)
        kwargs['validation'] = deepcopy(self.validation[step])
        kwargs['overview_id'] = self._step_id(self.boundaries[step], step)
        if kwargs.get('work_path') is not None:
            kwargs['work_path'] = _series_work_path(kwargs['work_path'], step, include_rank=not self.preload_data_modules)
        return kwargs

    def _expand_config(self, config):
        config = deepcopy(config)
        if config['type'] == 'map':
            configs = self._expand_map_config(config)
        else:
            configs = [config]

        if self.iterations is not None:
            for c in configs:
                if c['type'] in ['random_spherical', 'random_radial_grouped'] and 'length' not in c:
                    c['length'] = self.iterations
        return configs

    def _expand_map_config(self, config):
        files = config.get('files')
        if files is None:
            return [config]
        if isinstance(files, list):
            configs = []
            for f in files:
                c = deepcopy(config)
                c['files'] = f
                c['step_id'] = self._files_id(f)
                configs.append(c)
            return configs

        expanded_files = {k: self._expand_file_value(v) for k, v in files.items()}
        series_lengths = [len(v) for v in expanded_files.values() if isinstance(v, list)]
        if len(series_lengths) == 0:
            return [config]

        n_steps = max(series_lengths)
        assert all(length in (1, n_steps) for length in series_lengths), \
            f'Inconsistent number of files in spherical map config {config.get("id", "")}'

        configs = []
        for step in range(n_steps):
            c = deepcopy(config)
            c['files'] = {k: v[step if len(v) > 1 else 0] if isinstance(v, list) else v
                          for k, v in expanded_files.items()}
            c['step_id'] = self._files_id(c['files'])
            configs.append(c)
        return configs

    @staticmethod
    def _expand_file_value(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and re.search(r'\[\[[^\[\]]+\]\]', value):
            return value
        if isinstance(value, str) and glob.has_magic(value):
            files = sorted(glob.glob(value))
            assert len(files) > 0, f'No files found for pattern {value}'
            return files
        return value

    @staticmethod
    def _resolve_placeholders(value, context):
        if isinstance(value, dict):
            return {k: SphericalSeriesDataModule._resolve_placeholders(v, context)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [SphericalSeriesDataModule._resolve_placeholders(v, context) for v in value]
        if not isinstance(value, str):
            return value

        exact_match = re.fullmatch(r'\[\[([^\[\]]+)\]\]', value)
        if exact_match:
            return SphericalSeriesDataModule._resolve_placeholder(exact_match.group(1), context)

        def replace(match):
            resolved = SphericalSeriesDataModule._resolve_placeholder(match.group(1), context)
            return str(resolved)

        return re.sub(r'\[\[([^\[\]]+)\]\]', replace, value)

    @staticmethod
    def _resolve_placeholder(path, context):
        keys = path.split('.')
        resolved = context[keys[0]]
        for key in keys[1:]:
            if key == 'errors' and key not in resolved and 'files' in resolved:
                resolved = resolved['files']
                continue
            resolved = resolved[key]
        return resolved

    @staticmethod
    def _files_id(files):
        if 'Br' in files:
            return os.path.basename(files['Br']).split('.')[-3]
        return None

    @staticmethod
    def _step_id(boundaries, step):
        for config in boundaries:
            if 'step_id' in config and config['step_id'] is not None:
                return config['step_id']
            if config.get('type') == 'map' and 'files' in config:
                config_id = SphericalSeriesDataModule._files_id(config['files'])
                if config_id is not None:
                    return config_id
            if 'id' in config and config['id'] is not None:
                return config['id']
        return f'step_{step:06d}'

    @property
    def current_id(self):
        return self._step_id(self.boundaries[self.step], self.step)

    def activate_step(self, step):
        self.step = step
        if not self.preload_data_modules:
            self.data_modules = [None] * self.total_steps
        self._get_data_module(self.step)

    def _get_data_module(self, step):
        if self.data_modules[step] is None:
            self._evict_data_modules(keep_step=step)
            self.data_modules[step] = _load_spherical_data_module(
                (step, self.total_steps, self.boundaries[step], self.args, self._kwargs_for_step(step)))
        return self.data_modules[step]

    def _load_data_modules(self, data_module_workers):
        if not self.preload_data_modules:
            self._get_data_module(self.step)
            return

        n_workers = data_module_workers if data_module_workers is not None else self.kwargs.get(
            'num_workers', DEFAULT_NUM_WORKERS)
        n_workers = max(1, min(n_workers, self.total_steps))

        print(f'Loading data modules... (total: {self.total_steps}, workers: {n_workers})')
        worker_args = [(step, self.total_steps,
                        boundaries if step == self.step else _without_overview_plots(boundaries),
                        self.args, self._kwargs_for_step(step))
                       for step, boundaries in enumerate(self.boundaries)]
        if n_workers == 1:
            for args in worker_args:
                step = args[0]
                self.data_modules[step] = _load_spherical_data_module(args)
            return

        n_workers = max(1, min(n_workers, len(worker_args)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for args, data_module in zip(worker_args, pool.map(_load_spherical_data_module, worker_args)):
                step = args[0]
                self.data_modules[step] = data_module

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

    def train_dataloader(self):
        return self._get_data_module(self.step).train_dataloader()

    def val_dataloader(self):
        return self._get_data_module(self.step).val_dataloader()
