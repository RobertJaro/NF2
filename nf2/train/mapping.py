from nf2.data.dataset import SphereSlicesDataset, SlicesDataset, SphereDataset, CubeDataset
from nf2.loader.fits import FITSMapModule
from nf2.loader.spherical import SphericalSliceDataset
from nf2.train.callback import SphericalSlicesCallback, SlicesCallback, MetricsCallback, BoundaryCallback


def load_callbacks(data_module):
    callbacks = []
    Mm_per_ds = data_module.config['Mm_per_ds']
    G_per_dB = data_module.config['G_per_dB']
    for validation_dataset_key in data_module.validation_dataset_mapping.values():
        ds = data_module.validation_datasets[validation_dataset_key]
        if isinstance(ds, SphereSlicesDataset):
            callback = SphericalSlicesCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SlicesDataset):
            callback = SlicesCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SphereDataset) or isinstance(ds, CubeDataset):
            callback = MetricsCallback(validation_dataset_key, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SphericalSliceDataset) or isinstance(ds, FITSMapModule):
            callback = BoundaryCallback(validation_dataset_key, ds.cube_shape, G_per_dB)
        else:
            raise NotImplementedError(f'Data set {type(ds)} not implemented for validation.')
        callbacks += [callback]
    return callbacks
