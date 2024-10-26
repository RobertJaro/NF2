from nf2.data.dataset import SphereSlicesDataset, SlicesDataset, SphereDataset, CubeDataset
from nf2.loader.base import MapDataset
from nf2.loader.fits import FITSDataset
from nf2.loader.spherical import SphericalSliceDataset
from nf2.train.callback import SphericalSlicesCallback, SlicesCallback, MetricsCallback, BoundaryCallback, \
    LosTrvAziBoundaryCallback, DisambiguationCallback


def load_callbacks(data_module, additional_callbacks=[]):
    callbacks = []
    Mm_per_ds = data_module.config['Mm_per_ds']
    G_per_dB = data_module.config['G_per_dB']
    for validation_dataset_key in data_module.validation_dataset_mapping.values():
        ds = data_module.validation_datasets[validation_dataset_key]
        for callback_config in additional_callbacks:
            ds_id = callback_config['ds_id']
            callback_type = callback_config['type']
            if ds_id == validation_dataset_key:
                if callback_type == 'disambiguation':
                    callback = DisambiguationCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
                else:
                    raise NotImplementedError(f'Callback type {callback_type} not implemented')
                callbacks += [callback]
        if isinstance(ds, SphereSlicesDataset):
            callback = SphericalSlicesCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SlicesDataset):
            callback = SlicesCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SphereDataset) or isinstance(ds, CubeDataset):
            callback = MetricsCallback(validation_dataset_key, G_per_dB, Mm_per_ds)
        elif isinstance(ds, SphericalSliceDataset) or isinstance(ds, FITSDataset):
            callback = BoundaryCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif isinstance(ds, MapDataset):
            if ds.los_trv_azi:
                callback = LosTrvAziBoundaryCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
            else:
                callback = BoundaryCallback(validation_dataset_key, ds.cube_shape, G_per_dB, Mm_per_ds)
        else:
            raise NotImplementedError(f'Data set {type(ds)} not implemented for validation.')
        callbacks += [callback]
    return callbacks
