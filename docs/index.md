# NF2

NF2 is a framework for neural magnetic-field extrapolation that combines observational constraints with physics-informed training. The repository supports both cartesian and spherical workflows and is being organized around a shared execution, configuration, and export surface.

## Repository Aims

NF2 is intended to support:

- active-region extrapolations from vector magnetograms
- spherical and synoptic workflows for larger-scale magnetic structure
- time-series runs that reuse previous model state
- evaluation and export pipelines for downstream physical analysis

## Refactor Goals

The current repository direction is:

- one canonical YAML config schema
- one shared runner for single and series workflows
- separate geometry implementations behind explicit adapters
- one shared output and export layer
- a clearer documentation and evaluation surface

## Where To Start

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Configuration](configuration.md)
- [Geometry Workflows](geometries.md)
- [Coordinates and Units](coordinates.md)

## Read the Docs Readiness

This repository is prepared for static documentation hosting with MkDocs.

- RTD config: `.readthedocs.yml`
- local docs dependencies: `docs/requirements.txt`
- local strict build: `make docs`
- local live preview: `make docs-serve`
