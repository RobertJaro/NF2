# Configuration Catalog

This page summarizes the real configuration families currently present in `config/`.

## `13229`

- Files: 5
- Geometry: cartesian
- Modes: mostly single, one series
- Typical model: `vector_potential`
- Typical inputs: `fits`, `sharp` sequence
- Aim: SHARP AR 13229 workflows and time evolution

## `13664`

- Files: 4
- Geometry: cartesian
- Modes: single and series
- Typical model: `vector_potential`
- Typical inputs: SHARP series-oriented setups
- Aim: NOAA 13664 sequence studies

## `analytical`

- Files: 2
- Geometry: cartesian
- Modes: single
- Typical model: `vector_potential`
- Typical inputs: `analytical`
- Aim: Low and Lou reference-field extrapolation and synthetic validation

## `dkist`

- Files: 2
- Geometry: cartesian
- Typical models: `b`
- Typical inputs: `fits`, `los`
- Aim: DKIST/HMI embedded workflows

## `hinode`

- Files: 11
- Geometry: cartesian
- Typical model: `vector_potential`
- Typical inputs: `numpy`
- Aim: Hinode regional extrapolation and comparisons

## `magnetostatic`

- Files: 7
- Geometry: cartesian
- Typical models: `magneto_static`, `b`
- Typical inputs: `muram_cube`, `muram_pressure`
- Aim: magnetostatic and pressure-aware experiments

## `muram_example`

- Files: 2
- Geometry: cartesian
- Modes: single and series
- Typical model: `vector_potential`
- Typical inputs: `muram_slice`
- Aim: example MURaM-driven runs

## `nlte`

- Files: 5
- Geometry: cartesian
- Typical models: `vector_potential`, `gauged_vector_potential`
- Typical inputs: `los_trv_azi`, `fld_inc_azi`
- Aim: NLTE / multi-height ambiguity-aware workflows

## `scaling`

- Files: 2
- Geometry: cartesian
- Typical models: `vector_potential`, `gauged_vector_potential`
- Typical inputs: `fits`
- Aim: scaling studies and encoding/normalization experiments

## `sharp`

- Files: 5
- Geometry: cartesian
- Modes: mostly single, one series
- Typical models: `vector_potential`, `b`, `vector_potential_scaled`
- Typical inputs: `fits`, `sharp`, `numpy`
- Aim: standard SHARP training workflows

## `spherical`

- Files: 20
- Geometry: spherical
- Modes: mostly single, one series
- Typical models: `vector_potential`, `gauged_vector_potential`
- Typical inputs: `map`, `random_spherical`, `pfss_boundary`
- Typical validation: `map`, `sphere`, `spherical_slices`
- Aim: full-disk, synoptic, and spherical AR studies

## `sst`

- Files: 2
- Geometry: cartesian
- Typical model: `vector_potential`
- Typical inputs: `los_trv_azi`
- Aim: SST observational workflows

## `topology`

- Files: 23
- Geometry: cartesian
- Typical model: `vector_potential`
- Typical inputs: `muram_slice`, `los_trv_azi`, `fld_inc_azi`, `sharp`
- Aim: topology-focused and multi-slice experiments

## How To Use This Catalog

Use this page to find the right family quickly, then read:

- [Configuration](configuration.md)
- [Dataset Catalog](dataset_catalog.md)
- [Geometry Workflows](geometries.md)
