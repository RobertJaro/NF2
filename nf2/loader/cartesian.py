import glob
import os
from copy import copy, deepcopy

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from sunpy.map import Map

from nf2.data.dataset import RandomCoordinateDataset, CubeDataset, SlicesDataset
from nf2.data.loader import load_potential_field_boundary
from nf2.loader.base import TensorsDataset, BaseDataModule, MapDataset
from nf2.loader.muram import MURaMDataset, MURaMCubeDataset, MURaMPressureDataset


class CartesianDataModule(BaseDataModule):

    def __init__(self, train_configs, work_directory, valid_configs=None,
                 boundary_config=None, random_config=None,
                 Mm_per_ds=.36 * 320, G_per_dB=2500, z_range=None, validation_batch_size=2 ** 15, log_shape=False,
                 batch_size=int(2 ** 12), validation_ds_per_pixel=1 / 128, **kwargs):
        # wrap data if only one slice is provided
        train_configs = train_configs if isinstance(train_configs, list) else [train_configs]
        # boundary dataset
        training_datasets = {}
        for i, config in enumerate(train_configs):
            config = deepcopy(config)
            ds_id = config.pop('ds_id', f'train_{i + 1:02d}')
            config['batch_size'] = config.pop('batch_size', batch_size) # default batch size
            dataset = self.init_dataset(**config,
                                        Mm_per_ds=Mm_per_ds, G_per_dB=G_per_dB,
                                        work_directory=work_directory)
            training_datasets[ds_id] = dataset

        bottom_boundary_dataset = list(training_datasets.values())[0]

        # random sampling dataset
        coord_range = bottom_boundary_dataset.coord_range
        z_range = [0, 100] if z_range is None else z_range
        z_range_arr = np.array([z_range]) / Mm_per_ds
        coord_range = np.concatenate([coord_range, z_range_arr], axis=0)
        random_config = random_config if random_config is not None else {}
        random_batch_size = random_config.pop('batch_size', batch_size)
        random_dataset = RandomCoordinateDataset(coord_range, batch_size=random_batch_size, **random_config)

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

        training_datasets['random'] = random_dataset

        # top and side boundaries
        boundary_config = boundary_config if boundary_config is not None else \
            {'type': 'potential', 'strides': 4}
        boundary_batch_size = boundary_config.pop('batch_size', batch_size)
        boundary_type = boundary_config.pop('type')
        if boundary_type == 'none':
            pass
        elif boundary_type == 'potential':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            boundary_ds = PotentialBoundaryDataset(bz=bz,
                                                   height_pixel=coord_range[2, -1] / ds_per_pixel,
                                                   ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                   work_directory=work_directory,
                                                   batch_size=boundary_batch_size, **boundary_config)
            training_datasets['potential'] = boundary_ds
        elif boundary_type == 'potential_top':
            bz = bottom_boundary_dataset.bz
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            boundary_ds = PotentialTopBoundaryDataset(bz=bz,
                                                      height_pixel=coord_range[2, -1] / ds_per_pixel,
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
            valid_configs.append({'type': "cube", 'ds_id': 'cube',})
            valid_configs.append({'type': "slices", 'ds_id': 'slices',})
        for i, config in enumerate(valid_configs):
            config = deepcopy(config)
            ds_id = config.pop('ds_id', f'valid_{i + 1:02d}')
            config['batch_size'] = config.pop('batch_size', validation_batch_size)
            dataset = self.init_dataset(**config,
                                        Mm_per_ds=Mm_per_ds, G_per_dB=G_per_dB,
                                        shuffle=False, filter_nans=False,
                                        work_directory=work_directory, plot=False,
                                        ds_per_pixel=validation_ds_per_pixel, coord_range=coord_range)
            validation_datasets[ds_id] = dataset

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

        super().__init__(training_datasets, validation_datasets, config, **kwargs)

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
        elif type == 'slices':
            return SlicesDataset(**kwargs)
        elif type == 'cube':
            return CubeDataset(**kwargs)
        else:
            raise ValueError(f'Unknown boundary type: {type}. Supported types: '
                             f'fits, los_trv_azi, los, sharp, fld_inc_azi, numpy, muram_slice, muarm_cube')


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

    def __init__(self, fits_path, mask_path=None,
                 bin=1, slice=None, load_map=True, **kwargs):
        file_los = fits_path['B_los']
        file_trv = fits_path['B_trv']
        file_azi = fits_path['B_azi']

        if load_map:
            los_map, trv_map, azi_map = Map(file_los), Map(file_trv), Map(file_azi)
            los_map = process_map(los_map, slice, bin)
            trv_map = process_map(trv_map, slice, bin)
            azi_map = process_map(azi_map, slice, bin)
            los_data = los_map.data
            trv_data = trv_map.data
            azi_data = azi_map.data
            wcs = los_map.wcs
            if mask_path is not None:
                mask_map = Map(mask_path)
                mask_map = process_map(mask_map, slice, bin)
                mask = mask_map.data
                los_data[mask] = 0  # np.nan
                trv_data[mask] = 0  # np.nan
                azi_data[mask] = 0  # np.nan
        else:
            los_data = fits.getdata(file_los)
            trv_data = fits.getdata(file_trv)
            azi_data = fits.getdata(file_azi)
            if mask_path is not None:
                mask = fits.getdata(mask_path)
                mask = np.array(mask, dtype=bool)
                los_data[mask] = 0  # np.nan
                trv_data[mask] = 0  # np.nan
                azi_data[mask] = 0  # np.nan
            if slice:
                los_data = los_data[slice[0]:slice[1], slice[2]:slice[3]]
                trv_data = trv_data[slice[0]:slice[1], slice[2]:slice[3]]
                azi_data = azi_data[slice[0]:slice[1], slice[2]:slice[3]]
            if bin > 1:
                los_data = block_reduce(los_data, (bin, bin), func=np.mean)
                trv_data = block_reduce(trv_data, (bin, bin), func=np.mean)
                azi_data = block_reduce(azi_data, (bin, bin), func=np.mean)
            wcs = None

        b = np.stack([los_data, trv_data, np.pi - azi_data]).transpose()

        super().__init__(b=b, wcs=wcs, los_trv_azi=True, **kwargs)


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


class CartesianSeriesDataModule(CartesianDataModule):

    def __init__(self, fits_paths, error_paths=None, slice_type='fits', *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.slice_type = slice_type
        self.fits_paths = copy(fits_paths)
        self.error_paths = copy(error_paths) if error_paths is not None else [None] * len(fits_paths)

        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]

        self.initialized = True  # only required for first iteration
        super().__init__({'type': self.slice_type,
                          'fits_path': self.fits_paths[0],
                          'error_path': self.error_paths[0]}, *args, **kwargs)

    def train_dataloader(self):
        # skip reload if already initialized - for initial epoch
        if self.initialized:
            self.initialized = False
            print('Currently loaded:', self.current_id)
            return super().train_dataloader()
        # update ID
        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]
        # re-initialize
        super().__init__(train_configs={'type': self.slice_type, 'fits_path': self.fits_paths[0],
                                        'error_path': self.error_paths[0]},
                         *self.args, **self.kwargs)
        # continue with next file in list
        del self.fits_paths[0]
        del self.error_paths[0]
        print('Currently loaded:', self.current_id)
        return super().train_dataloader()


class SHARPDataset(FITSDataset):

    def __init__(self, **kwargs):
        super().__init__(Mm_per_pixel=.36, **kwargs)


class NumpyDataset(MapDataset):

    def __init__(self, data_path, **kwargs):
        data = np.load(data_path)
        bx = data['bx']
        by = data['by']
        bz = data['bz']
        bx_err = data.get('b_err', None)
        by_err = data.get('by_err', None)
        bz_err = data.get('bz_err', None)
        b = np.stack([bx, by, bz], axis=-1)
        if bx_err is not None and by_err is not None and bz_err is not None:
            b_err = np.stack([bx_err, by_err, bz_err], axis=-1)
        else:
            b_err = None

        super().__init__(b=b, b_err=b_err, **kwargs)


class PotentialBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, ds_per_pixel, G_per_dB, strides=2, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides,
                                                         progress=False)
        coords = coords * ds_per_pixel
        b_err = b_err / G_per_dB
        b = b / G_per_dB

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)


class PotentialTopBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, ds_per_pixel, G_per_dB, strides=2, batch_size=2 ** 12, **kwargs):
        coords, b_err, b = load_potential_field_boundary(bz, height_pixel, strides,
                                                         only_top=True, progress=False)
        coords = coords * ds_per_pixel
        b_err = b_err / G_per_dB
        b = b / G_per_dB

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)


def process_map(map, slice, bin):
    if slice:
        map = map.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                         top_right=u.Quantity((slice[1], slice[3]), u.pixel))
    if bin > 1:
        map = map.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean)
    return map
