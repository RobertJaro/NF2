# Full YAML Reference

This page is generated from `nf2.reference` and mirrors the public v0.4 YAML schema.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| path | str | ./runs/nf2 | Directory for checkpoints, logs, and extrapolation_result.nf2. |
| work_path | str \| null | path/work | Optional scratch directory for preprocessed arrays. |
| logging | dict | {} | Options passed to the Lightning/W&B logger. |
| data.geometry | cartesian \| spherical | required | Extrapolation geometry. |
| data.normalization.Mm_per_ds | float | 100 | Length represented by one model coordinate unit. |
| data.normalization.Gauss_per_dB | float | 1000 | Magnetic-field strength represented by one model field unit. |
| data.boundaries | list[dict] | required | Boundary observations, analytical fields, or maps. |
| data.boundaries[].Mm_per_pixel | float | dataset default | Spatial sampling for this boundary in Mm per pixel. |
| data.boundaries[].coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate assigned to the boundary center, in Mm. |
| data.validation | list[dict] | geometry default | Validation grids and plotting datasets. |
| data.sampler | dict | height sampler | Cartesian physics sampling dataset. |
| data.samplers | list[dict] | random_radial_grouped | Spherical physics sampling datasets. |
| data.potential_boundary | dict | FFT potential | Cartesian potential boundary data. Use type: none to disable. |
| data.z_range | list[float] | loader default | Cartesian height range in Mm where supported by the loader. |
| data.max_radius | float | loader default | Spherical outer radius in solar radii where supported by the loader. |
| data.iterations | int | loader default | Number of random sampler batches for Cartesian training. |
| model.field | vector_potential \| b | vector_potential | Field representation. |
| model.network.type | siren | siren | Only SIREN networks are supported. |
| model.network.hidden_dim | int | 256 cartesian, 512 spherical | SIREN hidden width. |
| model.network.layers | int | model default | Number of SIREN layers. |
| model.network.w0 | float | model default | SIREN frequency scale for hidden layers. |
| model.network.w0_initial | float | model default | SIREN frequency scale for the first layer. |
| training.epochs | int | 10 | Number of Lightning epochs. |
| training.optimizer.start | float | 5e-4 | Initial learning rate. |
| training.optimizer.end | float | 5e-5 | Final learning rate. |
| training.optimizer.iterations | int | 100000 | Learning-rate schedule length. |
| training.trainer | dict | {} | Additional Lightning Trainer keyword arguments. |
| losses[].type | str | geometry default | Loss implementation name. |
| losses[].name | str | type | Stable logging and scaling identifier. |
| losses[].weight | float \| schedule | required when explicit | Loss weight. The legacy lambda key is not accepted. |
| losses[].datasets | str \| list[str] | loss default | Dataset ids used by the loss. |
| loss_scaling | list[dict] | geometry default | Spatial scaling modules for selected losses. |
| callbacks | list[dict] | geometry default | Validation plots and metrics logged during training. |
| transforms | list[dict] | [] | Optional coordinate/field transforms applied to datasets. |
| meta_path | str | none | Optional previous NF2 state used by series runs. |
