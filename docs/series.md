# Series Runs

Series runs reuse the same public YAML schema as single extrapolations, but file placeholders usually point to glob patterns instead of single files. A series config tells NF2 how to build one boundary set per time step, then train each step while carrying forward state from the previous extrapolation.

A series run requires an initial single extrapolation first. Run the first time step with `nf2-extrapolate`, then pass that first run's `last.ckpt` as `--meta_path` when starting the full series. The meta path is the start point for the series and lets each subsequent time step continue from a trained NF2 state, including optimizer state, instead of starting cold.

## How The YAML Is Configured

The important difference from a single-run YAML file is `meta_path` and pattern-based file placeholders:

```yaml
path: "<<run_path>>"
work_path: "<<work_path>>"
meta_path: "<<meta_path>>"
data:
  geometry: cartesian
  boundaries:
    - id: photosphere
      type: los_trv_azi
      files:
        B_los: "<<photosphere_B_los_pattern>>"
        B_trv: "<<photosphere_B_trv_pattern>>"
        B_azi: "<<photosphere_B_azi_pattern>>"
```

When `nf2-extrapolate-series` loads the config, each pattern is expanded and sorted. At step `i`, NF2 uses the `i`th file from every matching component sequence. Every component sequence for a boundary must have the same length, and filenames should sort chronologically.

You can write patterns directly in a local YAML file:

```yaml
path: ./runs/multi_height_series
work_path: ./runs/multi_height_series/work
meta_path: ./runs/multi_height_initial/last.ckpt
data:
  geometry: cartesian
  boundaries:
    - id: photosphere
      type: los_trv_azi
      files:
        B_los: ./data/photosphere/*B_los.fits
        B_trv: ./data/photosphere/*B_trv.fits
        B_azi: ./data/photosphere/*B_azi.fits
```

Use placeholders when the same config should run on different active regions, instruments, or file systems.

## Step 1: Run The Initial Extrapolation

Use the matching single-run config for the first frame in the sequence:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/photosphere/20240101_000000_B_azi.fits"
```

Wait until this run writes `./runs/multi_height_initial/last.ckpt`.

## Step 2: Start The Full Series

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --work_path "./runs/multi_height_series/work" \
  --meta_path "./runs/multi_height_initial/last.ckpt" \
  --photosphere_B_los_pattern "./data/photosphere/*B_los.fits" \
  --photosphere_B_trv_pattern "./data/photosphere/*B_trv.fits" \
  --photosphere_B_azi_pattern "./data/photosphere/*B_azi.fits"
```

The `--meta_path` value must point to the completed first extrapolation's `last.ckpt` file. Keep that first run available until the series has started successfully.

Each component pattern in a boundary must expand to the same number of files. NF2 pairs files by sorted order, so use filenames that sort chronologically and consistently across components.

Series runs reuse `work_path/data_module.pkl` when it already exists, which avoids rebuilding every per-step data module after an interrupted run. Add `--reload` to the `nf2-extrapolate-series` command when you changed the input configuration or want to rebuild that saved state.

The example series configs use `check_val_every_n_epoch: 10`, so validation callbacks and W&B validation logging run every 10th dataset. The per-step `.nf2` result files are still saved at the end of every training epoch/dataset.

Multi-height extrapolations can be used as series runs when all boundary heights provide matching file sequences. First create the single-run meta path from one complete multi-height extrapolation, then start the full multi-height series with glob patterns for every photospheric and chromospheric component.

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
  --config "examples/configs/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea_377_initial" \
  --work_path "./runs/sharp_cea_377_initial/work" \
  --Br "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits" \
  --Bt "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits" \
  --Bp "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits" \
  --Br_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits" \
  --Bt_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits" \
  --Bp_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
```

Then start the SHARP CEA series by filling the placeholders in `examples/configs/cartesian/sharp_cea_series.yaml`:

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/sharp_cea_series.yaml" \
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

Then run the series with glob patterns:

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
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

## Spherical Series Runs

NF2 can run spherical sequences when the spherical boundary file entries are glob patterns or file lists. Use `examples/configs/spherical/hmi_full_disk.yaml` for the initial extrapolation and `examples/configs/spherical/hmi_full_disk_series.yaml` for the sequence.

As with Cartesian series, a spherical series needs the completed first extrapolation's `last.ckpt` as `meta_path`.

### 1. Run The First Spherical Extrapolation

Use the single-run guide in [Spherical extrapolations](spherical.md) for the first time step:

```bash
nf2-extrapolate \
  --config "examples/configs/spherical/hmi_full_disk.yaml" \
  --run_path "./runs/spherical_initial" \
  --work_path "./runs/spherical_initial/work" \
  --wandb_project "nf2" \
  --run_name "Spherical HMI initial" \
  --full_disk_Br "./data/hmi_spherical/full_disk/20240510_000000.Br.fits" \
  --full_disk_Bt "./data/hmi_spherical/full_disk/20240510_000000.Bt.fits" \
  --full_disk_Bp "./data/hmi_spherical/full_disk/20240510_000000.Bp.fits" \
  --full_disk_Br_err "./data/hmi_spherical/full_disk/20240510_000000.Br_err.fits" \
  --full_disk_Bt_err "./data/hmi_spherical/full_disk/20240510_000000.Bt_err.fits" \
  --full_disk_Bp_err "./data/hmi_spherical/full_disk/20240510_000000.Bp_err.fits" \
  --synoptic_Br "./data/hmi_spherical/synoptic/2283.Br.fits" \
  --synoptic_Bt "./data/hmi_spherical/synoptic/2283.Bt.fits" \
  --synoptic_Bp "./data/hmi_spherical/synoptic/2283.Bp.fits"
```

### 2. Use The Series Config

Use the series config directly and fill its `<<...>>` placeholders from the command line. Every glob must match either one shared file or the same number of time steps as the other series components.

Within a spherical series config, `[[dataset.path.to.value]]` references are resolved after dataset files are expanded. For example, `[[full_disk.files.Br]]` points to the current time step's full-disk `Br` file and is not a command-line override.

The series template validates and logs every 10th dataset by default while still saving one `.nf2` file per dataset.

### 3. Run The Series

```bash
nf2-extrapolate-series \
  --config "examples/configs/spherical/hmi_full_disk_series.yaml" \
  --run_path "./runs/spherical_series" \
  --work_path "./runs/spherical_series/work" \
  --meta_path "./runs/spherical_initial/last.ckpt" \
  --wandb_project "nf2" \
  --run_name "Spherical HMI series" \
  --full_disk_Br_pattern "./data/hmi_spherical/full_disk/*.Br.fits" \
  --full_disk_Bt_pattern "./data/hmi_spherical/full_disk/*.Bt.fits" \
  --full_disk_Bp_pattern "./data/hmi_spherical/full_disk/*.Bp.fits" \
  --full_disk_Br_err_pattern "./data/hmi_spherical/full_disk/*.Br_err.fits" \
  --full_disk_Bt_err_pattern "./data/hmi_spherical/full_disk/*.Bt_err.fits" \
  --full_disk_Bp_err_pattern "./data/hmi_spherical/full_disk/*.Bp_err.fits" \
  --synoptic_Br_pattern "./data/hmi_spherical/synoptic/*.Br.fits" \
  --synoptic_Bt_pattern "./data/hmi_spherical/synoptic/*.Bt.fits" \
  --synoptic_Bp_pattern "./data/hmi_spherical/synoptic/*.Bp.fits" \
  --fits_reference_Br "./data/hmi_spherical/reference/20240510_000000.Br.fits"
```

### 4. Export A Finished Series

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 1.44 \
  --height_range 0 100 \
  --metrics j alpha free_energy \
  --overwrite
```

```bash
nf2-export "./runs/spherical_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/spherical_series/exports" \
  --metrics j alpha free_energy \
  --overwrite
```

If a series step fails, inspect that time step's input files first. Mismatched component counts, missing error maps, or non-chronological filenames are the most common setup issues.

## Python API

Use `nf2.run_series(...)` to launch a series from Python. The same public schema is accepted, with glob patterns in the file fields:

```python
import nf2

nf2.run_series(
    path="./runs/multi_height_series",
    work_path="./runs/multi_height_series/work",
    meta_path="./runs/multi_height_initial/last.ckpt",
    data={
        "geometry": "cartesian",
        "boundaries": [
            {
                "id": "photosphere",
                "type": "los_trv_azi",
                "Mm_per_pixel": 0.09,
                "files": {
                    "B_los": "./data/photosphere/*B_los.fits",
                    "B_trv": "./data/photosphere/*B_trv.fits",
                    "B_azi": "./data/photosphere/*B_azi.fits",
                },
                "ambiguous_azimuth": True,
                "load_map": False,
            }
        ],
        "z_range": [0, 100],
    },
    training={"reload_dataloaders_every_n_epochs": 1, "check_val_every_n_epoch": 10},
)
```

After the series finishes, load or export individual `.nf2` files with the same output helpers used for single runs.
