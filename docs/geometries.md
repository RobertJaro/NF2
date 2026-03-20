# Geometry Workflows

NF2 supports two geometry families with a shared runner and export surface.

## Cartesian

Typical use cases:

- SHARP active-region extrapolations
- DKIST/Hinode/SST regional workflows
- MURaM local-box comparisons
- analytical Low and Lou reference-field experiments
- topology experiments

Common characteristics:

- local rectangular volume
- boundary conditions defined on one or more lower or interior slices
- validation datasets often include `cube` and `slices`

Common dataset types:

- `fits`
- `sharp`
- `los`
- `los_trv_azi`
- `fld_inc_azi`
- `muram_slice`
- `muram_cube`
- `analytical`

## Spherical

Typical use cases:

- full-disk extrapolations
- synoptic maps
- global magnetic energy and current studies

Common characteristics:

- global or regional spherical domain
- full-disk or synoptic map boundaries
- validation datasets often include `map`, `sphere`, and `spherical_slices`

Common dataset types:

- `map`
- `pfss_boundary`
- `random_spherical`
- `random_radial_grouped`
- `sphere`
- `spherical_slices`

## Shared Framework, Separate Implementations

The shared core handles:

- config loading
- run orchestration
- checkpointing
- export routing

The geometry packages still own:

- data-module implementation
- geometry-specific sampling
- geometry-specific boundary semantics
- geometry-specific output adapters
