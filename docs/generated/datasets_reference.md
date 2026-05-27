# Dataset And Sampler Reference

Dataset entries are used under `data.boundaries`, `data.validation`, `data.sampler`, and `data.samplers`.

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
