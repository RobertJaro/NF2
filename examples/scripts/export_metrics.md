# Export And Quality Metrics

Use these commands after training has produced one or more `.nf2` files.

## Export A Cartesian Result

```bash
nf2-export "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/sharp_cea_377/exports/field.vtk" \
  --Mm_per_pixel 1.44 \
  --height_range 0 80 \
  --metrics j alpha free_energy
```

Use `--format npz` for Python analysis or `--format hdf5` for a portable scientific data file. VTK output is useful for ParaView.

## Export A Spherical Result

```bash
nf2-export "./runs/spherical_hmi/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/spherical_hmi/exports/field.vtk" \
  --metrics j alpha free_energy
```

## Export A Series

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 1.44 \
  --height_range 0 100 \
  --metrics j alpha free_energy \
  --overwrite
```

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

The printed metrics include mean and RMS `divB`, mean and RMS `divB/B`, current-weighted `theta_J`, `sigma_J`/`CWsin`, total magnetic energy `E_tot`, Cartesian FFT free magnetic energy `E_free`, and `E_free/E_tot`.
