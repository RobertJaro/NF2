# Neural Network Force-Free Magnetic Field Extrapolation - NF2

[![Documentation Status](https://readthedocs.org/projects/nf2/badge/?version=latest)](https://nf2.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/nf2.svg)](https://pypi.org/project/nf2/)
[![Python versions](https://img.shields.io/pypi/pyversions/nf2.svg)](https://pypi.org/project/nf2/)
[![License](https://img.shields.io/pypi/l/nf2.svg)](LICENSE)

<img src="https://github.com/RobertJaro/NF2/blob/main/images/logo.jpg" width="150" height="150">

NF2 is a Python framework for neural non-linear force-free magnetic-field extrapolations. It supports Cartesian and spherical geometries, single-boundary and multi-height observations, extrapolation series, standard NLFF quality metrics, and exports for scientific analysis and visualization.

The framework is designed for solar-physics workflows with HMI/SHARP, full-disk, synoptic, and benchmark data, while keeping the training, export, and evaluation interfaces consistent across geometries.

## Highlights

- Unified Cartesian and spherical extrapolation workflows
- YAML-based configuration with explicit defaults
- SIREN-based neural-field model for all geometries
- Single-run and series extrapolations
- HMI SHARP, full-disk, and synoptic download helpers
- VTK, NumPy, and metric exports
- Standard NLFF quality metrics through `nf2-metrics`
- Sphinx documentation configured for ReadTheDocs

## Installation

NF2 supports Python 3.11 and 3.12.

Install from PyPI after release:

```bash
python -m pip install nf2
```

For development or source installs:

```bash
git clone https://github.com/RobertJaro/NF2.git
cd NF2
python -m pip install -e ".[wandb,jsoc,pfss,docs,dev]"
```

The recommended conda environment for local development is:

```bash
conda env create -f environment.yml
conda activate nf2
```

See [INSTALL.md](INSTALL.md) for source builds, optional dependencies, and the conda package recipe.

## Quick Start

Run an extrapolation from a YAML configuration:

```bash
nf2-extrapolate \
  --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path "./runs/ar377" \
  --work_path "/scratch/ar377" \
  --Br "/data/Br.fits" \
  --Bt "/data/Bt.fits" \
  --Bp "/data/Bp.fits"
```

Download HMI SHARP data:

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir "./data/sharp" \
  --email "you@example.org" \
  --sharp_num 377 \
  --t_start "2011-02-15T00:00:00"
```

Export a trained extrapolation:

```bash
nf2-export \
  "path/to/extrapolation_result.nf2" \
  --format vtk \
  --out "path/to/field.vtk" \
  --Mm_per_pixel 0.72 \
  --metrics j alpha free_energy_fft
```

Print standard NLFF quality metrics:

```bash
nf2-metrics \
  "path/to/extrapolation_result.nf2" \
  --Mm_per_pixel 0.72 \
  --height_range 0 80
```

Run a time series or parameter series:

```bash
nf2-extrapolate-series --config "path/to/series.yaml"
```

## Python API

```python
import nf2

out = nf2.load("path/to/extrapolation_result.nf2")
cube = out.load_cube(Mm_per_pixel=0.72, metrics=["j", "alpha"])
```

Direct helpers are also available:

```python
from nf2 import CartesianOutput, SphericalOutput, run, run_series
```

## Examples

Example configurations live in [examples/configs](examples/configs):

- `cartesian/sharp_cea.yaml`
- `cartesian/minimal_fits.yaml`
- `cartesian/auto_disambiguation.yaml`
- `cartesian/multi_height.yaml`
- `cartesian/multi_height_series.yaml`
- `spherical/full_disk_synoptic.yaml`
- `benchmark/analytical_case1.yaml`
- `benchmark/analytical_case2.yaml`

Notebook workflows live in [examples/notebooks](examples/notebooks) for SHARP CEA, Cartesian series, spherical HMI, and analytical benchmark runs. Command-line examples for downloads, extrapolations, exports, metrics, and series runs are collected in [examples/scripts/README.md](examples/scripts/README.md).

## Documentation

The full documentation is available at [nf2.readthedocs.io](https://nf2.readthedocs.io/).

The documentation includes:

- YAML configuration reference
- Cartesian and spherical workflows
- Dataset, sampler, and normalization options
- Download, extrapolation, export, and metric commands
- Python API reference
- Example notebook descriptions
- Publication list

## Visualization

NF2 results can be exported to VTK and visualized with ParaView:

![NF2 field visualization in ParaView](images/paraview.jpeg)

## Publications

Core NF2 method and tool-development papers:

- Jarolim, Thalmann, Veronig, and Podladchikova 2023, *Nature Astronomy*  
  "Probing the solar coronal magnetic field with physics-informed neural networks."  
  DOI: [10.1038/s41550-023-02030-9](https://doi.org/10.1038/s41550-023-02030-9)
- Jarolim, Tremblay, Rempel, Molnar, Veronig, Thalmann, and Podladchikova 2024, *The Astrophysical Journal Letters*  
  "Advancing solar magnetic field extrapolations through multiheight magnetic field measurements."  
  DOI: [10.3847/2041-8213/ad2450](https://doi.org/10.3847/2041-8213/ad2450)
- Jarolim, Veronig, Purkhart, Zhang, and Rempel 2024, *The Astrophysical Journal Letters*  
  "Magnetic field evolution of the solar active region 13664."  
  DOI: [10.3847/2041-8213/ad8914](https://doi.org/10.3847/2041-8213/ad8914)
- da Silva Santos, Dunnington, Jarolim, Danilovic, and Criscuoli 2025, *The Astrophysical Journal*  
  "Magnetic Reconnection in a Compact Magnetic Dome: Chromospheric Emissions and High-velocity Plasma Flows."  
  DOI: [10.3847/1538-4357/adcf23](https://doi.org/10.3847/1538-4357/adcf23)

See [docs/publications.md](docs/publications.md) for selected applications and related PINN NLFFF work.

## License

NF2 is released under the [GPL-3.0 license](LICENSE).
