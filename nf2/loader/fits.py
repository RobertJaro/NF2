import glob
import os
from copy import copy

import numpy as np
import wandb
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map

from nf2.data.dataset import RandomCoordinateDataset, CubeDataset, SlicesDataset
from nf2.data.loader import load_potential_field_data
from nf2.loader.base import TensorsDataset, BaseDataModule


class FITSDataModule(BaseDataModule):

    def __init__(self, fits_path, work_directory, error_path=None, fits_config=None, boundary_config=None,
                 random_config=None,
                 Mm_per_ds=.36 * 320, G_per_dB=2500, max_height=100, validation_batch_size=2 ** 15, log_shape=False,
                 **kwargs):
        # boundary dataset
        fits_config = fits_config if fits_config is not None else {}
        boundary_dataset = self.init_boundary_dataset(fits_path=fits_path, error_path=error_path,
                                                      Mm_per_ds=Mm_per_ds, G_per_dB=G_per_dB,
                                                      work_directory=work_directory, **fits_config)

        # random sampling dataset
        coord_range = boundary_dataset.coord_range
        z_range = np.array([[0, max_height / Mm_per_ds]])
        coord_range = np.concatenate([coord_range, z_range], axis=0)
        random_config = random_config if random_config is not None else {}
        random_dataset = RandomCoordinateDataset(coord_range, **random_config)

        ds_per_pixel = boundary_dataset.ds_per_pixel

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

        training_datasets = {'boundary': boundary_dataset, 'random': random_dataset}

        # top and side boundaries
        boundary_config = boundary_config if boundary_config is not None else {'type': 'potential', 'strides': 4}
        if boundary_config['type'] == 'potential':
            bz = Map(fits_path['Bp']).data.transpose()
            bz = np.nan_to_num(bz, nan=0)  # replace nans with 0
            potential_dataset = PotentialBoundaryDataset(bz=bz, height_pixel=max_height / (ds_per_pixel * Mm_per_ds),
                                                         ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                         work_directory=work_directory,
                                                         strides=boundary_config['strides'])
            training_datasets['potential'] = potential_dataset

        # validation datasets
        cube_dataset = CubeDataset(coord_range, batch_size=validation_batch_size)
        validation_boundary_dataset = self.init_boundary_dataset(fits_path=fits_path, error_path=error_path,
                                                                 Mm_per_ds=Mm_per_ds, G_per_dB=G_per_dB,
                                                                 shuffle=False, filter_nans=False,
                                                                 work_directory=work_directory,
                                                                 batch_size=validation_batch_size, plot=False,
                                                                 **fits_config)
        validation_slices_dataset = SlicesDataset(coord_range, ds_per_pixel, n_slices=10,
                                                  batch_size=validation_batch_size)

        validation_datasets = {'validation_boundary': validation_boundary_dataset, 'cube': cube_dataset,
                               'slices': validation_slices_dataset}

        config = {'type': 'cartesian',
                  'Mm_per_ds': Mm_per_ds, 'G_per_dB': G_per_dB, 'max_height': max_height,
                  'coord_range': coord_range, 'ds_per_pixel': ds_per_pixel, 'wcs': boundary_dataset.wcs}

        super().__init__(training_datasets, validation_datasets, config, **kwargs)

    def init_boundary_dataset(self, **kwargs):
        return FITSMapModule(**kwargs)


class SHARPDataModule(FITSDataModule):

    def init_boundary_dataset(self, **kwargs):
        return SHARPMapModule(**kwargs)


class FITSMapModule(TensorsDataset):

    def __init__(self, fits_path, error_path=None,
                 G_per_dB=2500, Mm_per_pixel=0.36, Mm_per_ds=.36 * 320,
                 bin=1, slice=None, height_mapping=None, plot=True, **kwargs):
        self.ds_per_pixel = (Mm_per_pixel * bin) / Mm_per_ds

        file_p = fits_path['Bp']
        file_t = fits_path['Bt']
        file_r = fits_path['Br']
        p_map, t_map, r_map = Map(file_p), Map(file_t), Map(file_r)
        p_map = self._process(p_map, slice, bin)
        t_map = self._process(t_map, slice, bin)
        r_map = self._process(r_map, slice, bin)


        b = np.stack([p_map.data, -t_map.data, r_map.data]).transpose() / G_per_dB
        coords = np.stack(np.mgrid[:b.shape[0], :b.shape[1], :1], -1).astype(np.float32) * self.ds_per_pixel
        coords = coords[:, :, 0, :]

        self.coord_range = np.array([[coords[..., 0].min(), coords[..., 0].max()],
                                     [coords[..., 1].min(), coords[..., 1].max()]])

        self.cube_shape = coords.shape[:-1]
        self.wcs = r_map.wcs

        tensors = {'b_true': b, 'coords': coords}

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
            p_error_map = self._process(p_error_map, slice, bin)
            t_error_map = self._process(t_error_map, slice, bin)
            r_error_map = self._process(r_error_map, slice, bin)
            b_err = np.stack([p_error_map.data, t_error_map.data, r_error_map.data]).transpose() / G_per_dB
            tensors['b_err'] = b_err

            if plot:
                self._plot_B_error(b * G_per_dB, b_err * G_per_dB, coords)
        else:
            if plot:
                self._plot_B(b * G_per_dB, coords)

        if height_mapping is not None:
            z = height_mapping['z']
            z_min = height_mapping['z_min'] if 'z_min' in height_mapping else 0
            z_max = height_mapping['z_max'] if 'z_max' in height_mapping else 0

            coords[..., 2] = z / Mm_per_ds
            ranges = np.zeros((*coords.shape[:-1], 2))
            ranges[..., 0] = z_min / Mm_per_ds
            ranges[..., 1] = z_max / Mm_per_ds
            tensors['height_range'] = ranges

        tensors = {k: v.reshape((-1, *v.shape[2:])).astype(np.float32) for k, v in tensors.items()}

        super().__init__(tensors, **kwargs)

    def _plot_B(self, b, coords):
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        im = axs[0, 0].imshow(b[..., 0].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[0, 0].set_title('Bp')
        im = axs[0, 1].imshow(coords[..., 0].T, origin='lower')
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[0, 1].set_title('Coordinate x')
        #
        im = axs[1, 0].imshow(b[..., 1].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[1, 0].set_title('Bt')
        im = axs[1, 1].imshow(coords[..., 1].T, origin='lower')
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[1, 1].set_title('Coordinate y')
        #
        im = axs[2, 0].imshow(b[..., 2].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[2, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[2, 0].set_title('Br')
        im = axs[2, 1].imshow(coords[..., 2].T, origin='lower')
        divider = make_axes_locatable(axs[2, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[2, 1].set_title('Coordinate z')
        wandb.log({'FITS data': wandb.Image(plt)})
        plt.close(fig)

    def _plot_B_error(self, b, b_err, coords):
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        im = axs[0, 0].imshow(b[..., 0].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[0, 0].set_title('Bp')
        im = axs[0, 1].imshow(b_err[..., 0].T, origin='lower', cmap='Reds', norm=LogNorm(vmin=1, vmax=1000))
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[0, 1].set_title('Bp error')
        im = axs[0, 2].imshow(coords[..., 0].T, origin='lower')
        divider = make_axes_locatable(axs[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[0, 2].set_title('Coordinate x')
        #
        im = axs[1, 0].imshow(b[..., 1].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[1, 0].set_title('Bt')
        im = axs[1, 1].imshow(b_err[..., 1].T, origin='lower', cmap='Reds', norm=LogNorm(vmin=1, vmax=1000))
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[1, 1].set_title('Bt error')
        im = axs[1, 2].imshow(coords[..., 1].T, origin='lower')
        divider = make_axes_locatable(axs[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[1, 2].set_title('Coordinate y')
        #
        im = axs[2, 0].imshow(b[..., 2].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
        divider = make_axes_locatable(axs[2, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[2, 0].set_title('Br')
        im = axs[2, 1].imshow(b_err[..., 2].T, origin='lower', cmap='Reds', norm=LogNorm(vmin=1, vmax=1000))
        divider = make_axes_locatable(axs[2, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[2, 1].set_title('Br error')
        im = axs[2, 2].imshow(coords[..., 2].T, origin='lower')
        divider = make_axes_locatable(axs[2, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        axs[2, 2].set_title('Coordinate z')
        wandb.log({'FITS data': wandb.Image(plt)})
        plt.close(fig)

    def _process(self, map, slice, bin):
        if slice:
            map = map.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                             top_right=u.Quantity((slice[1], slice[3]), u.pixel))
        if bin > 1:
            map = map.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean)
        return map


class SHARPSeriesDataModule(SHARPDataModule):

    def __init__(self, fits_paths, error_paths=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fits_paths = copy(fits_paths)
        self.error_paths = copy(error_paths) if error_paths is not None else [None] * len(fits_paths)

        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]

        super().__init__(self.fits_paths[0], error_paths=self.error_paths[0], *args, **kwargs)

    def train_dataloader(self):
        # update ID
        self.current_id = os.path.basename(self.fits_paths[0]['Br']).split('.')[-3]
        # re-initialize
        super().__init__(fits_path=self.fits_paths[0], error_path=self.error_paths[0],
                         *self.args, **self.kwargs)
        # continue with next file in list
        del self.fits_paths[0]
        del self.error_paths[0]
        return super().train_dataloader()


class SHARPMapModule(FITSMapModule):

    def __init__(self, **kwargs):
        super().__init__(Mm_per_pixel=.36, **kwargs)


class PotentialBoundaryDataset(TensorsDataset):

    def __init__(self, bz, height_pixel, ds_per_pixel, G_per_dB, strides=2, batch_size=2 ** 10, **kwargs):
        coords, b_err, b = load_potential_field_data(bz, height_pixel, strides,
                                                     progress=False)
        coords = coords * ds_per_pixel
        b_err = b_err / G_per_dB
        b = b / G_per_dB

        super().__init__({'b_true': b, 'b_err': b_err, 'coords': coords}, batch_size=batch_size, **kwargs)
