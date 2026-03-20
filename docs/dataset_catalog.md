# Dataset Catalog

This page summarizes the dataset types currently used by NF2 configs.

## Cartesian Dataset Types

### `fits`

Standard vector magnetogram input with `Br`, `Bt`, `Bp`, plus optional error maps.

Typical keys:

- `fits_path`
- `error_path`
- `slice`
- `bin`
- `Mm_per_pixel`
- `load_map`

### `sharp`

Series-oriented SHARP input resolved from a data directory or glob pattern.

Typical use:

- in `data.sequence.frames` for cartesian series runs

### `los`

Line-of-sight magnetic input. Used when only the LOS field is available.

### `los_trv_azi`

Line-of-sight, transverse, and azimuth representation. Common in SST/NLTE workflows.

### `fld_inc_azi`

Field strength, inclination, azimuth representation. Used in some SHARP/topology configurations.

### `numpy`

Numpy-based boundary data used in some Hinode workflows.

### `analytical`

Synthetic Low and Lou-style boundary data generated on the fly from the analytical reference field.

Typical keys:

- `case`
- `z_index`
- `field_scale`
- optional field overrides such as `resolution`, `bounds`, `psi`, and `l`

### `muram_slice`

2D MURaM slice input, often used for multi-height or topology experiments.

### `muram_cube`

MURaM cube input for volumetric comparisons and magnetostatic workflows.

### `muram_pressure`

Pressure boundary/cube input paired with MURaM data.

### Validation Helpers

These usually appear in `data.validation`:

- `cube`
- `slices`

## Spherical Dataset Types

### `map`

Spherical boundary map input from full-disk or synoptic products.

Typical keys:

- `files`
- `mask_configs`
- `insert`

### `pfss_boundary`

PFSS-derived boundary condition for spherical training workflows.

### `random_spherical`

Random interior spherical coordinate sampling.

### `random_radial_grouped`

Grouped random radial sampling for spherical interiors.

### `sphere`

Dense full-volume spherical validation dataset.

### `spherical_slices`

Radial spherical slice validation dataset.

## Notes

- User-facing latitude ranges are accepted in configs.
- Internally, spherical transforms use `(r, theta=colatitude, phi)`.
- For geometry-specific guidance, see [Geometry Workflows](geometries.md) and [Coordinates and Units](coordinates.md).
