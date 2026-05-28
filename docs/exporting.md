# Exporting

Use the unified export command:

```bash
nf2-export model.nf2 --format vtk --out model.vtk --metrics j alpha
nf2-export "series/*.nf2" --format hdf5 --out-dir exports --overwrite
nf2-export multi_height.nf2 --format height --out height_surfaces.npz
```

Supported formats:

- `vtk`
- `npz`
- `hdf5`
- `fits`
- `height` for learned multi-height surface mappings

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
Height-surface exports use `--Mm_per_pixel` and require a checkpoint with a height transform and `height_mapping` entries.

Available export metrics are listed in the generated [export and metrics reference](generated/export_metrics_reference.md).
