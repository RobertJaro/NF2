# Cartesian Series Runs

Series runs train a sequence of extrapolations. They require an initial single extrapolation first. That first run provides the `meta_path` start point, so the series continues from a trained NF2 state instead of starting each time step from random weights.

This guide has two paths:

- single-height SHARP CEA series, including the HMI download command
- multi-height Cartesian series, assuming you already have photospheric and chromospheric FITS files

## Single-Height SHARP CEA Series

First download a SHARP CEA vector time sequence from JSOC. Fill in your JSOC email and either a SHARP number or NOAA number.

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir "./data/sharp_cea_377" \
  --email "you@example.org" \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --t_end 2011-02-16T00:00:00 \
  --cadence 720s \
  --series sharp_cea_720s \
  --segments Br,Bp,Bt,Br_err,Bp_err,Bt_err
```

Use `--noaa_num` instead of `--sharp_num` when you want NF2 to resolve the HARP/SHARP number from a NOAA active-region number.

Check that every component downloaded the same number of files and that sorting is chronological:

```bash
ls ./data/sharp_cea_377/*.Br.fits | head
ls ./data/sharp_cea_377/*.Bt.fits | head
ls ./data/sharp_cea_377/*.Bp.fits | head
```

Run one complete single extrapolation for the first time step:

```bash
nf2-extrapolate \
  --config "nf2/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea_377_initial" \
  --work_path "./runs/sharp_cea_377_initial/work" \
  --Br "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits" \
  --Bt "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits" \
  --Bp "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits" \
  --Br_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits" \
  --Bt_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits" \
  --Bp_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
```

The `--Br_err`, `--Bt_err`, and `--Bp_err` arguments are optional. If you omit them, NF2 skips the error maps.

Then start the SHARP CEA series by filling the placeholders in `nf2/cartesian/sharp_cea_series.yaml`:

```bash
nf2-extrapolate-series \
  --config "nf2/cartesian/sharp_cea_series.yaml" \
  --run_path "./runs/sharp_cea_377_series" \
  --work_path "./runs/sharp_cea_377_series/work" \
  --meta_path "./runs/sharp_cea_377_initial/last.ckpt" \
  --Br_pattern "./data/sharp_cea_377/*.Br.fits" \
  --Bt_pattern "./data/sharp_cea_377/*.Bt.fits" \
  --Bp_pattern "./data/sharp_cea_377/*.Bp.fits" \
  --Br_err_pattern "./data/sharp_cea_377/*.Br_err.fits" \
  --Bt_err_pattern "./data/sharp_cea_377/*.Bt_err.fits" \
  --Bp_err_pattern "./data/sharp_cea_377/*.Bp_err.fits"
```

The `--Br_err_pattern`, `--Bt_err_pattern`, and `--Bp_err_pattern` arguments are optional. If you omit them, NF2 runs the series without error maps.

## Multi-Height Cartesian Series

Multi-height series assume you already have matching photospheric and chromospheric FITS files. The example config uses line-of-sight, transverse, and azimuth components for both heights:

```text
./data/multi_height/
  photosphere/
    20240101_000000_B_los.fits
    20240101_000000_B_trv.fits
    20240101_000000_B_azi.fits
  chromosphere/
    20240101_000000_B_los.fits
    20240101_000000_B_trv.fits
    20240101_000000_B_azi.fits
```

Use filenames that sort in chronological order. Every component pattern must match the same number of files:

```bash
ls ./data/multi_height/photosphere/*B_los.fits | head
ls ./data/multi_height/chromosphere/*B_los.fits | head
```

Run one complete single extrapolation for the first time step:

```bash
nf2-extrapolate \
  --config "nf2/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/multi_height/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/multi_height/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/multi_height/photosphere/20240101_000000_B_azi.fits" \
  --chromosphere_B_los "./data/multi_height/chromosphere/20240101_000000_B_los.fits" \
  --chromosphere_B_trv "./data/multi_height/chromosphere/20240101_000000_B_trv.fits" \
  --chromosphere_B_azi "./data/multi_height/chromosphere/20240101_000000_B_azi.fits"
```

Then run the series with glob patterns:

```bash
nf2-extrapolate-series \
  --config "nf2/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --work_path "./runs/multi_height_series/work" \
  --meta_path "./runs/multi_height_initial/last.ckpt" \
  --photosphere_B_los_pattern "./data/multi_height/photosphere/*B_los.fits" \
  --photosphere_B_trv_pattern "./data/multi_height/photosphere/*B_trv.fits" \
  --photosphere_B_azi_pattern "./data/multi_height/photosphere/*B_azi.fits" \
  --chromosphere_B_los_pattern "./data/multi_height/chromosphere/*B_los.fits" \
  --chromosphere_B_trv_pattern "./data/multi_height/chromosphere/*B_trv.fits" \
  --chromosphere_B_azi_pattern "./data/multi_height/chromosphere/*B_azi.fits"
```

If filename sorting is not enough to pair time steps correctly, use list notation in a local YAML config instead of glob placeholders. You can either put one list on each component key or make each list item one complete time step. Every component and every boundary height must have the same number of items:

```yaml
files:
  B_los:
    - ./data/multi_height/photosphere/20240101_000000_B_los.fits
    - ./data/multi_height/photosphere/20240101_001200_B_los.fits
  B_trv:
    - ./data/multi_height/photosphere/20240101_000000_B_trv.fits
    - ./data/multi_height/photosphere/20240101_001200_B_trv.fits
  B_azi:
    - ./data/multi_height/photosphere/20240101_000000_B_azi.fits
    - ./data/multi_height/photosphere/20240101_001200_B_azi.fits
```

The equivalent per-time-step form is:

```yaml
path: ./runs/multi_height_series
work_path: ./runs/multi_height_series/work
meta_path: ./runs/multi_height_initial/last.ckpt
data:
  geometry: cartesian
  boundaries:
    - id: photosphere
      type: los_trv_azi
      Mm_per_pixel: 0.09
      files:
        - B_los: ./data/multi_height/photosphere/20240101_000000_B_los.fits
          B_trv: ./data/multi_height/photosphere/20240101_000000_B_trv.fits
          B_azi: ./data/multi_height/photosphere/20240101_000000_B_azi.fits
        - B_los: ./data/multi_height/photosphere/20240101_001200_B_los.fits
          B_trv: ./data/multi_height/photosphere/20240101_001200_B_trv.fits
          B_azi: ./data/multi_height/photosphere/20240101_001200_B_azi.fits
      ambiguous_azimuth: true
      load_map: false
    - id: chromosphere
      type: los_trv_azi
      Mm_per_pixel: 0.09
      files:
        - B_los: ./data/multi_height/chromosphere/20240101_000000_B_los.fits
          B_trv: ./data/multi_height/chromosphere/20240101_000000_B_trv.fits
          B_azi: ./data/multi_height/chromosphere/20240101_000000_B_azi.fits
        - B_los: ./data/multi_height/chromosphere/20240101_001200_B_los.fits
          B_trv: ./data/multi_height/chromosphere/20240101_001200_B_trv.fits
          B_azi: ./data/multi_height/chromosphere/20240101_001200_B_azi.fits
      height_mapping: { z: 2.0, z_min: 0.0, z_max: 20.0 }
      ambiguous_azimuth: true
      load_map: false
  sampler:
    type: height
    batch_size: 16384
  potential_boundary:
    id: potential
    type: potential
    strides: 4
  iterations: 10000
  z_range: [0, 100]
training:
  reload_dataloaders_every_n_epochs: 1
  check_val_every_n_epoch: 10
```

Then run the copied config without per-component pattern arguments:

```bash
nf2-extrapolate-series --config "./my_multi_height_series.yaml"
```

List notation also accepts directories or glob dictionaries as list items, but explicit file dictionaries are the safest option when component names, timestamps, or observing cadences do not sort cleanly.

The series templates validate and log every 10th dataset by default while still saving one `.nf2` file per dataset.

## Export A Finished Series

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 0.72 \
  --height_range 0 100 \
  --metrics j alpha free_energy_fft \
  --overwrite
```

If a series step fails, inspect that time step's input files first. Mismatched component counts, missing error maps, or non-chronological filenames are the most common setup issues.
