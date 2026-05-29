# Quality Metrics

`nf2-metrics` prints a standard set of NLFF quality metrics for Cartesian and spherical checkpoints.

```bash
nf2-metrics "./runs/sharp_cea/extrapolation_result.nf2" \
  --Mm_per_pixel 1.44 \
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

## Cartesian Quality Metrics

```bash
nf2-metrics "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --Mm_per_pixel 1.44 \
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

The generated [export and metrics reference](generated/export_metrics_reference.md) lists all metric names.
