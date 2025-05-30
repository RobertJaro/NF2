from nf2.train.callback import SphericalSlicesCallback, SlicesCallback, MetricsCallback, BoundaryCallback, \
    LosTrvAziBoundaryCallback, DisambiguationCallback, PressureBoundaryCallback


def load_callbacks(callback_configs, data_module):
    callbacks = []
    Mm_per_ds = data_module.config['Mm_per_ds']
    G_per_dB = data_module.config['G_per_dB']

    for callback_config in callback_configs:
        ds_id = callback_config.pop('ds_id')
        callback_type = callback_config.pop('type')
        assert ds_id in data_module.validation_datasets, \
            f'Dataset {ds_id} not found in validation datasets. Check your configuration. Available datasets: {list(data_module.validation_datasets.keys())}'
        ds = data_module.validation_datasets[ds_id]

        if callback_type == 'disambiguation':
            callback = DisambiguationCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds, **callback_config)
        elif callback_type == 'spherical_slices':
            callback = SphericalSlicesCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif callback_type == 'slices':
            callback = SlicesCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif callback_type == 'metrics':
            callback = MetricsCallback(ds_id, G_per_dB, Mm_per_ds)
        elif callback_type == 'boundary':
            callback = BoundaryCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif callback_type == 'pressure_boundary':
            callback = PressureBoundaryCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds)
        elif callback_type == 'los_trv_azi_boundary':
            callback = LosTrvAziBoundaryCallback(ds_id, ds.cube_shape, G_per_dB, Mm_per_ds)
        else:
            raise NotImplementedError(f'Callback type {callback_type} not implemented.')
        callbacks.append(callback)
    return callbacks
