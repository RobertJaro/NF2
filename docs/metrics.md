# Quality Metrics

`nf2-metrics` prints a standard set of NLFF quality metrics for Cartesian and spherical checkpoints.

```bash
nf2-metrics "./runs/sharp_cea/extrapolation_result.nf2" \
  --Mm_per_pixel 0.72 \
  --height_range 0 80
```

For spherical checkpoints, choose the sample grid explicitly:

```bash
nf2-metrics "./runs/spherical/extrapolation_result.nf2" \
  --spherical_sampling 32 64 128 \
  --radius_range 1.0 1.3 \
  --latitude_range -60 60
```

`--spherical_sampling` is the number of samples in radius, latitude, and longitude. Larger values improve volume coverage and cost more memory/time.

Use the same sampling ranges when comparing runs. Metrics depend on the evaluated volume, so changing `--height_range`, `--radius_range`, grid spacing, or spherical sampling can change the reported values even when the checkpoint is the same.

## Cartesian Quality Metrics

```bash
nf2-metrics "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --Mm_per_pixel 0.72 \
  --height_range 0 80
```

## Spherical Quality Metrics

```bash
nf2-metrics "./runs/spherical_hmi/extrapolation_result.nf2" \
  --spherical_sampling 32 64 128 \
  --radius_range 1.0 1.3 \
  --latitude_range -60 60
```

The printed metrics include mean and RMS `divB`, mean and RMS `divB/B`, current-weighted `theta_J`, `sigma_J`/`CWsin`, total magnetic energy `E_tot`, Cartesian FFT free magnetic energy `E_free`, and `E_free/E_tot`. Spherical free energy is currently reported as `n/a` because the implemented free-energy estimate uses a Cartesian FFT potential-field reference.

Read the metrics as diagnostics rather than pass/fail thresholds:

- `mean_abs_divB`, `rms_divB`, `mean_abs_divB_over_B`, and `rms_divB_over_B` measure how close the field is to divergence-free. Lower is better when the same volume and sampling are used.
- `theta_J`, `sigma_J`, `CWsin`, `mean_force_free_residual`, and `rms_force_free_residual` measure current-field alignment and force-free consistency. Lower values indicate better alignment.
- `E_tot` is the total magnetic energy in the sampled volume. Compare it only across runs evaluated on the same physical domain.
- `E_free`, `E_pot`, and `E_free_over_E_tot` estimate the non-potential energy content for Cartesian checkpoints. For spherical checkpoints these free-energy fields are reported as `n/a`.
- `mean_B` and `max_B` are useful sanity checks for unit mistakes, unexpectedly weak fields, or ranges that include too much quiet volume.

For a new setup, run metrics on the analytical smoke test first, then on a small observational volume. If metrics keep improving at the end of training, increase `training.epochs` or `data.iterations`; if they are noisy or out of memory, reduce the evaluation grid or pass a smaller `--batch_size`.

The generated [export and metrics reference](generated/export_metrics_reference.md) lists all metric names.
