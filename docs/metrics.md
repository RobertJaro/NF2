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

The printed metrics include divergence, normalized divergence, current-weighted force-free angles, total magnetic energy, Cartesian FFT free energy, and field-strength summaries. Spherical free energy is currently reported as `n/a` because the implemented free-energy estimate uses a Cartesian FFT potential-field reference.

The generated [export and metrics reference](generated/export_metrics_reference.md) lists all metric names.
