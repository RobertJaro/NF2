import numpy as np

from nf2.data.dataset import RandomCoordinateDataset, CubeDataset, SlicesDataset
from nf2.loader.base import MapDataset, BaseDataModule
from nf2.loader.fits import PotentialBoundaryDataset


class MURaMDataModule(BaseDataModule):

    def __init__(self, slices, work_directory, boundary_config=None,
                 random_config=None,
                 Mm_per_ds=.36 * 320, G_per_dB=2500, max_height=100, validation_batch_size=2 ** 15, log_shape=False,
                 **kwargs):
        # boundary dataset
        slice_datasets = []
        for slice_config in slices:
            muram_dataset = MURaMSliceDataset(**slice_config, G_per_dB=G_per_dB, Mm_per_ds=Mm_per_ds, work_directory=work_directory)
            slice_datasets.append(muram_dataset)

        bottom_boundary_dataset = slice_datasets[0]

        # random sampling dataset
        coord_range = bottom_boundary_dataset.coord_range
        z_range = np.array([[0, max_height / Mm_per_ds]])
        coord_range = np.concatenate([coord_range, z_range], axis=0)
        random_config = random_config if random_config is not None else {}
        random_dataset = RandomCoordinateDataset(coord_range, **random_config)

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

        training_datasets = {}
        for i, dataset in enumerate(slice_datasets):
            training_datasets[f'boundary_{i + 1:02d}'] = dataset
        training_datasets['random'] = random_dataset

        # top and side boundaries
        boundary_config = boundary_config if boundary_config is not None else {'type': 'potential', 'strides': 8}
        if boundary_config['type'] == 'potential':
            sl, Nvar, shape, time = read_muram_slice(slices[0]['data_path'])
            bz = sl[5, :, :] * np.sqrt(4 * np.pi)
            potential_dataset = PotentialBoundaryDataset(bz=bz, height_pixel=max_height / (ds_per_pixel * Mm_per_ds),
                                                         ds_per_pixel=ds_per_pixel, G_per_dB=G_per_dB,
                                                         work_directory=work_directory,
                                                         strides=boundary_config['strides'])
            training_datasets['potential'] = potential_dataset

        # validation datasets
        cube_dataset = CubeDataset(coord_range, batch_size=validation_batch_size)

        validation_slice_datasets = []
        for slice_config in slices:
            muram_dataset = MURaMSliceDataset(**slice_config, G_per_dB=G_per_dB, Mm_per_ds=Mm_per_ds, work_directory=work_directory,
                                              shuffle=False, filter_nans=False, batch_size=validation_batch_size, plot=False)
            validation_slice_datasets.append(muram_dataset)
        validation_slices_dataset = SlicesDataset(coord_range, ds_per_pixel, n_slices=10,
                                                  batch_size=validation_batch_size)

        validation_datasets = {'cube': cube_dataset, 'slices': validation_slices_dataset}
        for i, dataset in enumerate(validation_slice_datasets):
            validation_datasets[f'validation_boundary_{i + 1:02d}'] = dataset

        config = {'type': 'cartesian',
                  'Mm_per_ds': Mm_per_ds, 'G_per_dB': G_per_dB, 'max_height': max_height,
                  'coord_range': coord_range, 'ds_per_pixel': ds_per_pixel}

        super().__init__(training_datasets, validation_datasets, config, **kwargs)


class MURaMSliceDataset(MapDataset):

    def __init__(self, data_path, *args, **kwargs):
        sl, Nvar, shape, time = read_muram_slice(data_path)

        bz = sl[5, :, :] * np.sqrt(4 * np.pi)
        bx = sl[6, :, :] * np.sqrt(4 * np.pi)
        by = sl[7, :, :] * np.sqrt(4 * np.pi)

        b = np.stack([bx, by, bz], axis=-1)

        super().__init__(b, Mm_per_pixel=0.192, *args, **kwargs)

def read_muram_slice(filepath):
    data = np.fromfile(filepath, dtype=np.float32)
    Nvar = data[0].astype(int)
    shape = tuple(data[1:3].astype(int))
    time = data[3]
    slice = data[4:].reshape([Nvar, shape[1], shape[0]]).swapaxes(1, 2)
    return slice, Nvar, shape, time
