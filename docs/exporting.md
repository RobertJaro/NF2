# Exporting

Use the unified export command:

```bash
nf2-export model.nf2 --format vtk --out model.vtk --metrics j alpha
nf2-export "series/*.nf2" --format hdf5 --out-dir exports --overwrite
```

Supported formats:

- `vtk`
- `npz`
- `hdf5`
- `fits`

For single files, use `--out`. For multiple inputs or glob patterns, use `--out-dir`.

```bash
nf2-export "runs/series/*.nf2" \
  --format hdf5 \
  --out-dir "runs/series/exports" \
  --Mm_per_pixel 1.44 \
  --height_range 0 80 \
  --metrics j alpha free_energy_fft \
  --overwrite
```

Cartesian exports accept `--Mm_per_pixel`, `--height_range`, `--x_range`, and `--y_range`. Spherical VTK export detects spherical checkpoints automatically.

Available export metrics are listed in the generated [export and metrics reference](generated/export_metrics_reference.md).
