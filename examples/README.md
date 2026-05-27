# NF2 Examples

This directory collects runnable NF2 examples for command-line runs, YAML configuration templates, and notebooks. The examples are intended to be copied, edited, and reused with local data paths.

Most YAML files use `<<...>>` placeholders so one template can be reused with different input files, scratch paths, and output directories. The placeholders can be filled from the command line with `nf2-extrapolate` or replaced directly in a local copy of the YAML file.

Start from the repository root:

```bash
conda activate nf2
export WANDB_MODE=offline
```

## Directory Overview

- [configs](configs): YAML templates for single runs, series runs, and analytical benchmarks.
- [scripts](scripts): command-line guides showing how to download data, run configs, export results, and compute metrics.
- [notebooks](notebooks): interactive examples for local Jupyter and Google Colab.

## Configuration Templates

The configuration templates live under [configs](configs). They follow the public NF2 configuration schema and can be used directly with the command-line tools.

### Benchmark Analytical Cases

Use these when you want a self-contained smoke test or a compact analytical NLFF benchmark. They generate Low & Lou fields internally and do not require external data.

- [configs/benchmark/analytical_case1.yaml](configs/benchmark/analytical_case1.yaml): analytical case 1 with compact batch sizes, `1000` sampler iterations per epoch, and `10` epochs.
- [configs/benchmark/analytical_case2.yaml](configs/benchmark/analytical_case2.yaml): analytical case 2 with the same benchmark training setup.

Guide: [scripts/benchmark.md](scripts/benchmark.md)

### Cartesian Single Runs

Use these for local Cartesian volumes above planar boundary maps.

- [configs/cartesian/minimal_fits.yaml](configs/cartesian/minimal_fits.yaml): minimal Cartesian run for plain Br/Bt/Bp FITS arrays. Uses `type: fits` and `load_map: false`.
- [configs/cartesian/sharp_cea.yaml](configs/cartesian/sharp_cea.yaml): SHARP CEA vector magnetogram run with map metadata.
- [configs/cartesian/auto_disambiguation.yaml](configs/cartesian/auto_disambiguation.yaml): Cartesian LOS/transverse/azimuth run with azimuth disambiguation support.
- [configs/cartesian/multi_height.yaml](configs/cartesian/multi_height.yaml): single multi-height Cartesian extrapolation with photospheric and chromospheric boundaries.

Guide: [scripts/cartesian.md](scripts/cartesian.md)

### Cartesian Series Runs

Use [configs/cartesian/multi_height_series.yaml](configs/cartesian/multi_height_series.yaml) for a time series of Cartesian multi-height extrapolations.

Series runs require a completed first single extrapolation. That first run provides the `meta_path` start point for the sequence, and the series then continues from that trained NF2 state.

Guide: [scripts/cartesian_series.md](scripts/cartesian_series.md)

### Spherical Single Runs

Use [configs/spherical/full_disk_synoptic.yaml](configs/spherical/full_disk_synoptic.yaml) for spherical HMI extrapolations that combine full-disk vector data with Carrington synoptic maps.

Guide: [scripts/spherical.md](scripts/spherical.md)

### Spherical Series Runs

The repository ships a single-run spherical template. For a spherical series, copy [configs/spherical/full_disk_synoptic.yaml](configs/spherical/full_disk_synoptic.yaml), replace single-file entries with glob patterns or file lists, and provide a completed first spherical extrapolation as `meta_path`.

Guide: [scripts/spherical_series.md](scripts/spherical_series.md)

## Notebooks

The notebooks are useful when you want a guided, editable run instead of command-line invocation.

- [notebooks/benchmark_analytical.ipynb](notebooks/benchmark_analytical.ipynb): analytical benchmark run and validation.
- [notebooks/sharp_cea_cartesian.ipynb](notebooks/sharp_cea_cartesian.ipynb): SHARP CEA download and Cartesian extrapolation.
- [notebooks/colab_sharp_cea.ipynb](notebooks/colab_sharp_cea.ipynb): Google Colab SHARP CEA example with NF2 installation steps.
- [notebooks/cartesian_series.ipynb](notebooks/cartesian_series.ipynb): Cartesian series example, including the initial run used as the series `meta_path`.
- [notebooks/spherical_hmi.ipynb](notebooks/spherical_hmi.ipynb): spherical HMI example.

## Export And Metrics

After training, use `nf2-export` to write VTK, NPZ, or HDF5 products, and `nf2-metrics` to evaluate extrapolation quality.

Guide: [scripts/export_metrics.md](scripts/export_metrics.md)

Typical quality checks include boundary agreement, `divB`, current-weighted `theta_J`, `sigma_J`/`CWsin`, total magnetic energy, and free magnetic energy.

## Common Usage Pattern

1. Choose the closest YAML template in [configs](configs).
2. Copy the template or pass placeholder values with `nf2-extrapolate`.
3. Set `run_path` to a writable output directory, usually under `runs/`.
4. Keep the produced `extrapolation_result.nf2`; it is the main input for exports, metrics, and series initialization.
5. For series runs, first train the initial single extrapolation and then use its `extrapolation_result.nf2` as `meta_path`.

Example placeholder invocation:

```bash
nf2-extrapolate \
  --config examples/configs/cartesian/minimal_fits.yaml \
  --run_path ./runs/minimal \
  --Br ./data/Br.fits \
  --Bt ./data/Bt.fits \
  --Bp ./data/Bp.fits
```

For one-off local runs, you can also replace placeholders directly in the YAML file with literal paths.

## Troubleshooting

- Use `WANDB_MODE=offline` or `WANDB_MODE=disabled` to avoid online W&B logging.
- Reduce sampler, boundary, or validation batch sizes when a run hits GPU out-of-memory errors.
- For large active regions, reduce the validation resolution or height range before changing the training setup.
- For series runs, check that every glob pattern matches the same number of files and that filenames sort chronologically.
