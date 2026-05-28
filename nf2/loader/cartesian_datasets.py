import glob
import os

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from sunpy.map import Map

from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.loader import load_potential_field_boundary
from nf2.data.dataset import TensorsDataset
from nf2.loader.base import MapDataset


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
