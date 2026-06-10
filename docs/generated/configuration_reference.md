# Full YAML Reference

This page is generated from `nf2.reference` and mirrors the public v0.4 YAML schema.

## General Run Keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| path | str | ./runs/nf2 | Directory for checkpoints, logs, and extrapolation_result.nf2. |
| work_path | str \| null | path/work | Optional scratch directory for preprocessed arrays. |
| logging | dict | {} | Options passed to the Lightning/W&B logger. |
| logging.project | str | logger default | W&B project name when W&B logging is enabled. |
| logging.name | str | logger default | Run name shown in W&B/log output. |
| meta_path | str | none | Optional previous NF2 state used by series runs. |

## Data And Geometry

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| data.geometry | cartesian \| spherical | required | Extrapolation geometry. |
| data.normalization.Mm_per_ds | float | 100 | Length represented by one model coordinate unit. |
| data.normalization.Gauss_per_dB | float | 1000 | Magnetic-field strength represented by one model field unit. |
| data.boundaries | list[dict] | required | Boundary observations, analytical fields, or maps. |
| data.boundaries[].id | str | boundary / generated | Stable dataset id used by losses, transforms, callbacks, and series references. |
| data.boundaries[].type | str | required | Dataset implementation name. See the dataset reference for supported types. |
| data.boundaries[].files | dict \| list[dict] \| glob | dataset dependent | Input file mapping or series file mapping for observational datasets. |
| data.boundaries[].errors | dict \| list[dict] | optional | Optional uncertainty-file mapping. Bundled templates skip unresolved error placeholders. |
| data.boundaries[].batch_size | int | data batch size | Per-dataset batch size override. |
| data.boundaries[].requires_jacobian | bool | dataset dependent | Whether this dataset must provide derivatives for losses or callbacks. |
| data.boundaries[].Mm_per_pixel | float | dataset default | Spatial sampling for this boundary in Mm per pixel. |
| data.boundaries[].coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate assigned to the boundary center, in Mm. |
| data.boundaries[].height_mapping | dict | none | Height metadata for multi-height Cartesian boundary datasets. |
| data.boundaries[].mask_configs | dict \| list[dict] | none | Spherical map masking/filtering configuration. |
| data.boundaries[].mask_configs.type | str | mask dependent | Mask/filter implementation name, for example `mu_filter` or `reference`. |
| data.boundaries[].mask_configs.min | float | mask dependent | Minimum value for simple threshold-style masks such as `mu_filter`. |
| data.boundaries[].mask_configs.file | str | required for reference masks | Reference map file used by spherical `reference` masks. |
| data.boundaries[].mask_configs.mu_filter | dict | optional | Nested mu-angle filter applied inside a spherical `reference` mask. |
| data.boundaries[].mask_configs.mu_filter.min | float | mask dependent | Minimum mu value for the nested reference-mask filter. |
| data.boundaries[].mask_configs[].type | str | mask dependent | Mask/filter implementation name when masks are provided as a list. |
| data.boundaries[].mask_configs[].min | float | mask dependent | Minimum value when list-form masks use `mu_filter`. |
| data.validation | list[dict] | geometry default | Validation grids and plotting datasets. |
| data.validation[].id | str | generated | Stable validation dataset id used by callbacks. |
| data.validation[].type | str | required | Validation dataset implementation name. |
| data.validation[].files | dict \| list[dict] \| glob | dataset dependent | Input file mapping for validation datasets. |
| data.validation[].errors | dict \| list[dict] | optional | Optional validation uncertainty-file mapping. |
| data.validation[].filter_nans | bool | dataset default | Whether to filter NaN samples in spherical map validation datasets. |
| data.validation[].shuffle | bool | dataset default | Whether to shuffle validation samples. |
| data.validation[].plot_overview | bool | dataset default | Whether to prepare overview plots for spherical map validation datasets. |
| data.validation[].plot_currents | bool | dataset default | Whether spherical slice validation plots include current-density views. |
| data.validation[].n_slices | int | dataset default | Number of slices for slice-based validation datasets. |
| data.validation[].mask_configs.type | str | mask dependent | Mask/filter implementation name for validation map datasets. |
| data.validation[].mask_configs.min | float | mask dependent | Minimum value for validation threshold-style masks such as `mu_filter`. |
| data.validation[].mask_configs[].type | str | mask dependent | Mask/filter implementation name when validation masks are provided as a list. |
| data.validation[].mask_configs[].min | float | mask dependent | Minimum value when list-form validation masks use `mu_filter`. |
| data.batch_size | int | 4096 spherical, 8192 cartesian | Default dataset batch size passed to loaders. |
| data.validation_batch_size | int | 16384 | Cartesian validation batch size. |
| data.validation_pixel_per_ds | float | 128 | Cartesian validation sampling density; larger values make coarser validation cubes. |
| data.sampler | dict | height sampler | Cartesian physics sampling dataset. |
| data.sampler.type | height \| default | height | Cartesian random sampler type. |
| data.sampler.batch_size | int | 16384 in examples | Cartesian random sampler batch size. |
| data.samplers | list[dict] | random_radial_grouped | Spherical physics sampling datasets. |
| data.samplers[].id | str | random | Spherical sampler id used by losses. |
| data.samplers[].type | random_radial_grouped \| random_spherical | random_radial_grouped | Spherical random sampler type. |
| data.samplers[].batch_size | int | data.batch_size | Spherical random sampler batch size. |
| data.potential_boundary | dict | FFT potential | Cartesian potential boundary data. Use type: none to disable. |
| data.potential_boundary.id | str | ignored | Accepted for readability; normalized away because the runtime id is `potential`. |
| data.potential_boundary.type | potential \| potential_top \| none | potential | Potential side/top boundary mode. |
| data.potential_boundary.strides | int | 4 | Downsample factor for potential boundary generation. |
| data.z_range | list[float] | loader default | Cartesian height range in Mm where supported by the loader. |
| data.max_radius | float | loader default | Spherical outer radius in solar radii where supported by the loader. |
| data.iterations | int | 10000 | Number of random sampler batches per epoch-like pass. |
| data.num_workers | int | 4 | Default PyTorch DataLoader workers for training and validation loaders. Series preloading uses this value unless data.data_module_workers is set. |
| data.train_num_workers | int | data.num_workers | PyTorch DataLoader workers for training loaders. |
| data.validation_num_workers | int | data.num_workers | PyTorch DataLoader workers for validation loaders. |
| data.prefetch_factor | int | 5 | Training DataLoader prefetch factor when workers are enabled. |
| data.persistent_workers | bool | true | Keep training DataLoader workers alive while a loader is active. |
| data.preload_data_modules | bool | true | For series runs, preload all step data modules up front instead of loading each step lazily. |
| data.data_module_workers | int | data.num_workers | Series-only multiprocessing workers used to preload per-step data modules. |

## Model

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| model.field | vector_potential \| scaled_vector_potential \| b | vector_potential | Field representation. |
| model.network.type | siren | siren | Only SIREN networks are supported. |
| model.network.hidden_dim | int | 256 cartesian, 512 spherical | SIREN hidden width. |
| model.network.layers | int | model default | Number of SIREN layers. |
| model.network.w0 | float | model default | SIREN frequency scale for hidden layers. |
| model.network.w0_initial | float | model default | SIREN frequency scale for the first layer. |
| model.radial_power | float | 2.0 | Radial power for `scaled_vector_potential`, applied as `(r / R_sun)^-radial_power`. |
| model.coordinate_radial_power | float | 4.0 | Radial power for compressing SIREN input coordinates in `scaled_vector_potential`, applied as `coords * (r / R_sun)^-coordinate_radial_power`. |
| model.base_radius | float | R_sun in model units | Reference radius for `scaled_vector_potential`; omit for spherical runs. |

## Training

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| training.epochs | int | 15 | Number of Lightning epochs. |
| training.optimizer.start | float | 5e-4 | Initial learning rate. |
| training.optimizer.end | float | 5e-5 | Final learning rate. |
| training.optimizer.iterations | int | 100000 | Learning-rate schedule length. |
| training.gradient_clip_val | float | 0.1 | Gradient clipping value passed to the Lightning Trainer unless overridden in training.trainer. |
| training.matmul_precision | str | medium | Torch matmul precision setting used before training. |
| training.reload_dataloaders_every_n_epochs | int | 1 for series | Series cadence for advancing to the next dataset. |
| training.check_val_every_n_epoch | int | 1 | Lightning validation cadence. Series examples use 10 to validate every 10th dataset. |
| training.val_check_interval | int \| float | Lightning default | Optional Lightning validation interval alias used by single-run training. |
| training.trainer | dict | {} | Additional Lightning Trainer keyword arguments. |
| training.trainer.* | any | Lightning default | Additional keyword passed through to `lightning.pytorch.Trainer`. |

## Losses And Scaling

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| losses | list[dict] | geometry default | Training objective terms. Omit to use geometry-specific defaults. |
| losses[].type | str | geometry default | Loss implementation name. Supported values are boundary, boundary_los_trv, boundary_azi, boundary_los_trv_azi, boundary_los, divergence, force_free, potential, weighted_height, height, NaNs, radial, min_height, energy_gradient, energy, sigma_j. |
| losses[].name | str | type | Stable logging and scaling identifier. |
| losses[].weight | float \| schedule | required when explicit | Loss weight. The legacy lambda key is not accepted. |
| losses[].weight.type | exponential \| linear \| step | exponential | Schedule type when `weight` is a mapping. |
| losses[].weight.start | float | required for exponential/linear | Initial scheduled loss weight. |
| losses[].weight.end | float | required for scheduled weights | Final scheduled loss weight. |
| losses[].weight.iterations | int | required for exponential/linear | Number of optimizer steps over which to change the weight. |
| losses[].weight.steps | int | required for step | Step interval for step schedules. |
| losses[].datasets | str \| list[str] | loss default | Dataset ids used by the loss. |
| losses[].ambiguous | bool | loss default | Enable ambiguity-aware behavior for supported azimuth/disambiguation losses. |
| loss_scaling | list[dict] | geometry default | Spatial scaling modules for selected losses. |
| loss_scaling[].type | exponential \| potential_fit \| b_height \| radial | geometry default | Loss-scaling module type. |
| loss_scaling[].name | str | type | Stable loss-scaling module name. |
| loss_scaling[].loss_ids | list[str] | required | Loss names to scale. |
| loss_scaling[].base_radius | float | 1.1 radial default | Solar-radius baseline for radial scaling. |
| loss_scaling[].max_radius | float \| null | none | Optional outer radius used to normalize radial scaling. |
| loss_scaling[].power | float | module default | Exponent used by b_height or potential_fit scaling. |

## Callbacks And Transforms

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| callbacks | list[dict] | geometry default | Validation plots and metrics logged during training. |
| callbacks[].type | boundary \| metrics \| slices \| spherical_slices \| fits_comparison \| disambiguation \| los_trv_azi_boundary | required | Callback implementation name. |
| callbacks[].dataset | str | required for most callbacks | Public alias for validation dataset id; normalized internally to `ds_id`. |
| callbacks[].name | str | callback default | Optional display/logging name for callbacks that support it. |
| callbacks[].plot | bool | true | For plotting callbacks, set false to keep scalar metrics but skip image rendering/logging. |
| callbacks[].component_labels | list[str] | callback default | Optional labels for vector-component plots. |
| transforms | list[dict] | [] | Optional coordinate/field transforms applied to datasets. |
| transforms[].type | height_range \| height \| optical_depth \| azimuth | required | Transform implementation name. |
| transforms[].datasets | str \| list[str] | required | Dataset ids to transform; normalized internally to `ds_id`. |
| transforms[].height_range | list[float] | required for height | Height range in Mm for learned height transforms. |
| transforms[].max_height | float | required for optical_depth | Maximum optical-depth transform height in Mm. |
| transforms[].max_log_optical_depth | float | -5 | Maximum log optical depth for optical-depth transforms. |
