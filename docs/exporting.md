# Exporting

Use the unified export command:

```bash
nf2-export model.nf2 --format vtk --out model.vtk --metrics j alpha
nf2-export "series/*.nf2" --format hdf5 --out-dir exports --overwrite
nf2-export multi_height.nf2 --format height --out height_surfaces.npz
```

Supported formats:

- `vtk` for Cartesian and spherical checkpoints
- `npz` for Cartesian checkpoints
- `hdf5` for Cartesian checkpoints
- `fits` for Cartesian checkpoints
- `height` for learned multi-height surface mappings

For single files, use `--out`. For multiple inputs or glob patterns, use `--out-dir`.

Every export includes the magnetic field evaluated from the NF2 checkpoint. The `--metrics` option adds derived quantities to the output. Metric names, default metrics, and scalar/vector array keys are shared between Cartesian and spherical exports. Use only the quantities needed for the next analysis step because derivative-heavy or field-line quantities can increase runtime and memory use.

## Export A Cartesian Result

```bash
nf2-export "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/sharp_cea_377/exports/field.vtk" \
  --Mm_per_pixel 0.72 \
  --height_range 0 80 \
  --metrics j alpha free_energy_fft
```

Use `--format npz` for Python analysis or `--format hdf5` for a portable scientific data file. VTK output is useful for ParaView.

## Export A Spherical Result

```bash
nf2-export "./runs/spherical_hmi/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/spherical_hmi/exports/field.vtk" \
  --metrics j alpha energy spherical_energy_gradient
```

Spherical checkpoints currently export through VTK only. Spherical VTK files include the magnetic field, spherical coordinate scalars, and any requested scalar or vector export quantities that match the sampled spherical grid.

## Export A Series

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 0.72 \
  --height_range 0 100 \
  --metrics j alpha free_energy_fft \
  --overwrite
```

Cartesian exports accept `--Mm_per_pixel`, `--height_range`, `--x_range`, and `--y_range`. Spherical VTK export detects spherical checkpoints automatically and accepts `--radius_range`, `--latitude_range`, `--longitude_range`, and `--pixels_per_solRad`.
Height-surface exports use `--Mm_per_pixel` and require a checkpoint with a height transform and `height_mapping` entries.

## Export Quantities

Pass one or more names after `--metrics`:

```bash
nf2-export "./runs/case/extrapolation_result.nf2" \
  --format npz \
  --out "./runs/case/field.npz" \
  --Mm_per_pixel 0.72 \
  --height_range 0 80 \
  --metrics j alpha energy free_energy_fft
```

Available derived quantities are:

| `--metrics` name | Exported array key | Notes |
| --- | --- | --- |
| `j` | `j` | Current-density magnitude `|J|`. |
| `j_vec` | `j_vec` | Current-density vector. |
| `alpha` | `alpha` | Force-free alpha, computed as `(J . B) / |B|^2`. |
| `b_nabla_bz` | `b_nabla_bz` | Vertical magnetic tension-related derivative. |
| `energy` | `energy` | Magnetic energy density in `erg / cm^3`. |
| `energy_gradient` | `energy_gradient` | Cartesian vertical magnetic-energy gradient. |
| `spherical_energy_gradient` | `spherical_energy_gradient` | Spherical radial magnetic-energy gradient. |
| `free_energy` | `free_energy` | Cartesian free magnetic energy density using the default potential-field method. |
| `free_energy_fft` | `free_energy_fft` | Free magnetic energy density using the Cartesian FFT potential field. |
| `free_energy_direct` | `free_energy_direct` | Free magnetic energy density using the direct potential-field method. |
| `magnetic_helicity` | `magnetic_helicity` | Magnetic helicity diagnostic; requires vector-potential output from compatible checkpoints. |
| `los_trv_azi` | `los_trv_azi` | LOS field, transverse-field magnitude, and azimuth components. |
| `squashing_factor` | `squashing_factor`, `twist` | Squashing factor and twist diagnostics; requires optional GPU/CuPy/FastQSL dependencies. |

Exported array keys now match the requested metric name wherever a metric produces one quantity. `squashing_factor` also writes `twist` because the field-line calculation naturally produces both diagnostics.
Use `--metrics j j_vec ...` when you want both the current-density magnitude and the full current-density vector in the exported file.

Available export quantities and quality metrics are also listed in the generated [export and metrics reference](generated/export_metrics_reference.md).
