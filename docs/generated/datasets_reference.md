# Dataset And Sampler Reference

Dataset entries are used under `data.boundaries`, `data.validation`, `data.sampler`, and `data.samplers`.

## Dataset Types

| Type | Role | Description |
| --- | --- | --- |
| analytical | benchmark boundary | Low & Lou analytical fields for case1/case2 benchmarks. |
| sharp | cartesian boundary | SHARP CEA Br/Bt/Bp FITS input, optionally with uncertainty maps. |
| fits | cartesian boundary | Generic Cartesian Br/Bt/Bp FITS files. |
| los_trv_azi | cartesian boundary | LOS/transverse/azimuth FITS input with automatic disambiguation support. |
| los | cartesian boundary | Line-of-sight-only Cartesian boundary input. |
| fld_inc_azi | cartesian boundary | Field strength/inclination/azimuth FITS input. |
| height | cartesian sampler | Random Cartesian volume samples for force-free and potential losses. |
| potential | cartesian boundary | Potential-field boundary generated from the photospheric boundary. |
| map | spherical boundary | Spherical Br/Bt/Bp map input for full-disk or synoptic maps. |
| random_radial_grouped | spherical sampler | Grouped radial samples in the spherical volume. |
| sphere | spherical validation | Spherical validation grid for field-quality callbacks. |
| spherical_slices | spherical validation | Radial or angular slices for spherical visualization callbacks. |

## `analytical`

Role: benchmark boundary.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| case | int \| str | 1 | Analytical Low & Lou case identifier. |
| bounds | list[float] | loader default | Physical Cartesian volume bounds for the benchmark field. |
| resolution | int \| list[int] | loader default | Sampling resolution used to generate the benchmark boundary. |
| batch_size | int | shared boundary batch size | Boundary samples per training batch. |

## `sharp`

Role: cartesian boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.Br | str | required | Radial magnetic-field FITS file. |
| files.Bt | str | required | Theta/transverse magnetic-field FITS file. |
| files.Bp | str | required | Phi/azimuthal magnetic-field FITS file. |
| errors.* | str | optional | Optional uncertainty FITS files, stored as `error_path` internally. |
| Mm_per_pixel | float | FITS/WCS default | Boundary plate scale in Mm per pixel. |
| coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate of the image center in Mm. |
| slice | slice \| null | null | Optional image subset. |
| batch_size | int | shared boundary batch size | Boundary samples per training or validation batch. |

## `fits`

Role: cartesian boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.Br | str | required | Radial/vertical magnetic-field FITS file. |
| files.Bt | str | required | Transverse x/theta component FITS file. |
| files.Bp | str | required | Transverse y/phi component FITS file. |
| Mm_per_pixel | float | dataset default | Boundary plate scale in Mm per pixel. |
| coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate of the image center in Mm. |
| batch_size | int | shared boundary batch size | Boundary samples per training or validation batch. |

## `los_trv_azi`

Role: cartesian boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.B_los | str | required | Line-of-sight magnetic-field FITS file. |
| files.B_trv | str | required | Transverse-field magnitude FITS file. |
| files.B_azi | str | required | Azimuth FITS file. |
| ambiguous_azimuth | bool | false | Train/evaluate both azimuth disambiguation branches when supported by the loss. |
| load_map | bool | dataset default | Load map metadata for coordinate conversion and plotting. |
| height_mapping | dict | none | Map this boundary to a height or learned height range, for multi-height runs. |
| Mm_per_pixel | float | dataset default | Boundary plate scale in Mm per pixel. |
| coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate of the image center in Mm. |
| batch_size | int | shared boundary batch size | Boundary samples per training or validation batch. |

## `los`

Role: cartesian boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.B_los | str | required | Line-of-sight magnetic-field FITS file. |
| Mm_per_pixel | float | dataset default | Boundary plate scale in Mm per pixel. |
| coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate of the image center in Mm. |
| batch_size | int | shared boundary batch size | Boundary samples per training or validation batch. |

## `fld_inc_azi`

Role: cartesian boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.B | str | required | Field-strength FITS file. |
| files.inc | str | required | Inclination FITS file. |
| files.azi | str | required | Azimuth FITS file. |
| Mm_per_pixel | float | dataset default | Boundary plate scale in Mm per pixel. |
| coordinate_center | list[float] \| dict | [0, 0] | Cartesian coordinate of the image center in Mm. |
| batch_size | int | shared boundary batch size | Boundary samples per training or validation batch. |

## `height`

Role: cartesian sampler.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| batch_size | int | 16384 in examples | Random volume samples per training batch. |
| z_sample | int | 128 | Number of heights sampled per xy location. |
| z_sampling_exponent | float | 2 | Bias random samples toward lower heights when greater than 1. |
| length | int \| null | data.iterations | Number of sampler batches per epoch-like pass. |
| requires_jacobian | bool | true | Required for force-free, divergence, and potential losses. |

## `potential`

Role: cartesian potential boundary.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| type | potential \| potential_top \| none | potential | Potential-field side/top boundary mode. |
| strides | int | 4 | Downsample factor for generating the potential boundary. |
| method | fft \| direct | fft | Potential-field solver. |
| batch_size | int | boundary batch size / 4 | Potential boundary samples per batch. |
| requires_jacobian | bool | false | Usually disabled for boundary matching. |

## `cube`

Role: cartesian validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| ds_per_pixel | float | 1 / validation_pixel_per_ds | Validation grid spacing in model units. |
| batch_size | int | validation_batch_size | Validation samples per batch. |
| requires_jacobian | bool | true | Needed for quality metrics that use derivatives. |

## `slices`

Role: cartesian validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| n_slices | int | 10 | Number of height slices to evaluate and plot. |
| batch_size | int | 4096 | Validation samples per batch. |
| requires_jacobian | bool | true | Needed for current-density plots. |

## `map`

Role: spherical boundary or validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| files.Br | str | required | Radial magnetic-field map. |
| files.Bt | str | required | Theta magnetic-field map. |
| files.Bp | str | required | Phi magnetic-field map. |
| errors.Br_err | str | optional | Radial uncertainty map, merged into the file map internally. |
| errors.Bt_err | str | optional | Theta uncertainty map, merged into the file map internally. |
| errors.Bp_err | str | optional | Phi uncertainty map, merged into the file map internally. |
| mask_configs | dict | optional | Optional masking/filtering, for example a `mu_filter`. |
| batch_size | int | data.batch_size or dataset default | Map samples per batch. |
| requires_jacobian | bool | false for boundary examples | Enable only when the dataset feeds derivative-based losses. |

## `random_radial_grouped`

Role: spherical sampler.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| batch_size | int | 16384 in examples | Random spherical volume samples per batch. |
| n_lat_lon_sample | int | 128 | Number of angular samples; `batch_size` must be divisible by this value. |
| radial_sampling_exponent | float | 1 | Bias radial samples when greater than 1. |
| latitude_range | list[float] | [-90, 90] | Latitude range in degrees unless `unit` is changed. |
| longitude_range | list[float] | [0, 360] | Longitude range in degrees unless `unit` is changed. |
| length | int \| null | data.iterations | Internal sampler length; public YAML should usually use data.iterations. |
| requires_jacobian | bool | true | Required for force-free, potential, and energy-gradient losses. |

## `sphere`

Role: spherical validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| resolution | int | 256 | Radial and longitude resolution of the validation sphere. |
| batch_size | int | 1024 | Validation samples per batch. |
| latitude_range | list[float] | [-90, 90] | Latitude range in degrees unless `unit` is changed. |
| longitude_range | list[float] | [0, 360] | Longitude range in degrees unless `unit` is changed. |
| requires_jacobian | bool | true | Needed for derivative-based quality metrics. |

## `spherical_slices`

Role: spherical validation.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| longitude_resolution | int | 256 | Longitude resolution for slice plots. |
| n_slices | int | 5 | Number of radial slices. |
| batch_size | int | 1024 | Validation samples per batch. |
| latitude_range | list[float] | [-90, 90] | Latitude range in degrees unless `unit` is changed. |
| longitude_range | list[float] | [0, 360] | Longitude range in degrees unless `unit` is changed. |
| requires_jacobian | bool | true | Needed for current-density plots. |
