# Examples

Examples in this repository currently come in three forms:

## 1. Real Config Families

The fastest examples are the actual configs under `config/`.

Good starting points:

- `config/analytical/case1.yaml`
- `config/analytical/case2.yaml`
- `config/sharp/377.yaml`
- `config/sharp/377_series.yaml`
- `config/spherical/377_spherical.yaml`
- `config/spherical/synoptic.yaml`

## 2. Notebooks

The `notebooks/` directory contains exploratory and demonstration notebooks, including:

- `01_Quickstart_Analytical_Extrapolation.ipynb`
- `02_Quickstart_SHARP_By_Date.ipynb`
- `03_Quickstart_Spherical_By_Date.ipynb`
- `04_Open_Sample_Checkpoint.ipynb`
- `05_Export_Checkpoint.ipynb`
- `06_Evaluate_Sample_Result.ipynb`
- `07_Open_Sample_HDF5.ipynb`

The click-and-go quickstarts are:

- `01_Quickstart_Analytical_Extrapolation.ipynb` for the built-in analytical reference field
- `02_Quickstart_SHARP_By_Date.ipynb` for SHARP download plus cartesian extrapolation
- `03_Quickstart_Spherical_By_Date.ipynb` for full-disk download plus spherical extrapolation

The supporting sample notebooks are:

- `04_Open_Sample_Checkpoint.ipynb` to inspect a sample NF2 checkpoint
- `05_Export_Checkpoint.ipynb` to export a checkpoint
- `06_Evaluate_Sample_Result.ipynb` to inspect evaluation workflows
- `07_Open_Sample_HDF5.ipynb` to inspect exported HDF5 output

## 3. Export Examples

Single result:

```bash
nf2-export --checkpoint /path/to/result.nf2 --format vtk
```

Spherical export with explicit ranges:

```bash
nf2-export --checkpoint /path/to/result.nf2 --format vtk \
  --radius_range 1.0 1.3 \
  --latitude_range -60 60 \
  --longitude_range 0 180 \
  --pixels_per_solRad 64
```

Series export:

```bash
nf2-export --series --checkpoint "/path/to/results/*.nf2" --format hdf5 --out_dir /path/to/exports
```
