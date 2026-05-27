# Running NF2 Examples

This directory documents the command-line workflow for the primary example configurations. The YAML files use `<<...>>` placeholders so the same template can be reused with local data paths, scratch paths, and active-region selections.

Run commands from an activated environment:

```bash
conda activate nf2
export WANDB_MODE=offline
```

## Benchmark Cases

The analytical benchmark examples do not need downloaded input data.

```bash
nf2-extrapolate \
  --config "examples/configs/benchmark/analytical_case1.yaml" \
  --run_path "./runs/benchmark/case1" \
  --work_path "./runs/benchmark/case1/work"

nf2-extrapolate \
  --config "examples/configs/benchmark/analytical_case2.yaml" \
  --run_path "./runs/benchmark/case2" \
  --work_path "./runs/benchmark/case2/work"
```

## SHARP CEA

Download a SHARP CEA vector magnetogram from JSOC:

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir "./data/sharp_cea_377" \
  --email "you@example.org" \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --t_end 2011-02-15T00:12:00 \
  --cadence 720s \
  --series sharp_cea_720s \
  --segments Br,Bp,Bt,Br_err,Bp_err,Bt_err
```

Run the SHARP CEA example configuration with the downloaded files:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea_377" \
  --work_path "./runs/sharp_cea_377/work" \
  --Br "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits" \
  --Bt "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits" \
  --Bp "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits"
```

For a minimal Cartesian run, use only the path and FITS placeholders:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/minimal_fits.yaml" \
  --run_path "./runs/sharp_cea_minimal" \
  --Br "./data/sharp_cea_377/Br.fits" \
  --Bt "./data/sharp_cea_377/Bt.fits" \
  --Bp "./data/sharp_cea_377/Bp.fits"
```

## Multi-Height Cartesian

Multi-height examples assume you already prepared matching photospheric and chromospheric FITS files. No data download command is provided for this workflow because the source instruments and preprocessing are project-specific.

Run a single multi-height extrapolation:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/multi_height/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/multi_height/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/multi_height/photosphere/20240101_000000_B_azi.fits" \
  --chromosphere_B_los "./data/multi_height/chromosphere/20240101_000000_B_los.fits" \
  --chromosphere_B_trv "./data/multi_height/chromosphere/20240101_000000_B_trv.fits" \
  --chromosphere_B_azi "./data/multi_height/chromosphere/20240101_000000_B_azi.fits"
```

Run a multi-height series after the first extrapolation has produced an initial `extrapolation_result.nf2`. Series configs use glob patterns; each component pattern must match the same number of files.

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --work_path "./runs/multi_height_series/work" \
  --meta_path "./runs/multi_height_initial/extrapolation_result.nf2" \
  --photosphere_B_los_pattern "./data/multi_height/photosphere/*B_los.fits" \
  --photosphere_B_trv_pattern "./data/multi_height/photosphere/*B_trv.fits" \
  --photosphere_B_azi_pattern "./data/multi_height/photosphere/*B_azi.fits" \
  --chromosphere_B_los_pattern "./data/multi_height/chromosphere/*B_los.fits" \
  --chromosphere_B_trv_pattern "./data/multi_height/chromosphere/*B_trv.fits" \
  --chromosphere_B_azi_pattern "./data/multi_height/chromosphere/*B_azi.fits"
```

## HMI Spherical

Download the full-disk vector field and convert it to spherical Br/Bt/Bp components through JSOC's `HmiB2ptr` processing step:

```bash
nf2-download \
  --source hmi_full_disk \
  --download_dir "./data/hmi_spherical/full_disk" \
  --email "you@example.org" \
  --t_start 2024-05-10T00:00:00 \
  --series B_720s
```

Download the Carrington synoptic vector maps:

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --carrington_rotation 2283 \
  --series b_synoptic \
  --segments Br,Bt,Bp
```

Run the spherical template by filling the file placeholders. The full-disk error placeholders can point to dedicated uncertainty maps; if those are not available, use the same field maps while preparing a proper uncertainty treatment for production runs.

```bash
nf2-extrapolate \
  --config "examples/configs/spherical/full_disk_synoptic.yaml" \
  --run_path "./runs/spherical_hmi" \
  --work_path "./runs/spherical_hmi/work" \
  --full_disk_Br "./data/hmi_spherical/full_disk/*Br.fits" \
  --full_disk_Bt "./data/hmi_spherical/full_disk/*Bt.fits" \
  --full_disk_Bp "./data/hmi_spherical/full_disk/*Bp.fits" \
  --full_disk_Br_err "./data/hmi_spherical/full_disk/*Br.fits" \
  --full_disk_Bt_err "./data/hmi_spherical/full_disk/*Bt.fits" \
  --full_disk_Bp_err "./data/hmi_spherical/full_disk/*Bp.fits" \
  --synoptic_Br "./data/hmi_spherical/synoptic/*Br.fits" \
  --synoptic_Bt "./data/hmi_spherical/synoptic/*Bt.fits" \
  --synoptic_Bp "./data/hmi_spherical/synoptic/*Bp.fits"
```

## Export After a Run

Export a Cartesian result:

```bash
nf2-export "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/sharp_cea_377/exports/field.vtk" \
  --Mm_per_pixel 1.44 \
  --height_range 0 80 \
  --metrics j alpha free_energy
```

Export a spherical result:

```bash
nf2-export "./runs/spherical_hmi/extrapolation_result.nf2" \
  --format vtk \
  --out "./runs/spherical_hmi/exports/field.vtk" \
  --metrics j alpha free_energy
```

## Quality Metrics

Compute the standard NLFF quality metrics for a Cartesian extrapolation:

```bash
nf2-metrics "./runs/sharp_cea_377/extrapolation_result.nf2" \
  --Mm_per_pixel 1.44 \
  --height_range 0 80
```

Compute metrics for a spherical extrapolation on a configurable spherical sample grid:

```bash
nf2-metrics "./runs/spherical_hmi/extrapolation_result.nf2" \
  --spherical_sampling 32 64 128 \
  --radius_range 1.0 1.3 \
  --latitude_range -60 60
```

The printed metrics include mean and RMS `divB`, mean and RMS `divB/B`, current-weighted `theta_J`, `sigma_J`/`CWsin`, total magnetic energy `E_tot`, Cartesian FFT free magnetic energy `E_free`, and `E_free/E_tot`.

Export a Cartesian series:

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 1.44 \
  --height_range 0 100 \
  --metrics j alpha free_energy \
  --overwrite
```
