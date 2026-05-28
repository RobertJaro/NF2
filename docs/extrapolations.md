# Extrapolations

NF2 extrapolations are usually run from YAML configuration files. The YAML file describes the run directory, input data, geometry, model, training settings, losses, callbacks, and export/evaluation datasets. The command-line tools load that YAML file, fill any placeholders, normalize the public schema, and then start the same Python training API used by scripts.

Choose the page that matches the geometry or run type:

- [Cartesian extrapolations](cartesian.md) cover local active-region boxes and multi-height Cartesian inputs.
- [Spherical extrapolations](spherical.md) cover full-disk and synoptic-map runs.
- [Analytical NLFF cases](analytical.md) provide fast benchmark and smoke-test runs.
- [Series runs](series.md) run time sequences after an initial single extrapolation has produced the starting NF2 state.

## YAML Configuration Files

A YAML configuration has a small set of top-level sections:

```yaml
path: ./runs/case
work_path: ./runs/case/work
logging:
  project: nf2
  name: "My extrapolation"
data:
  geometry: cartesian
  boundaries: []
training:
  epochs: 30
losses: []
callbacks: []
```

The required section is `data`. It must set `data.geometry` to `cartesian` or `spherical` and provide one or more boundary datasets. Everything else can be explicit or left to defaults. For a complete list of accepted keys, see the [Full YAML Reference](generated/configuration_reference.md) and [Dataset and Sampler Reference](generated/datasets_reference.md).

The command-line tools accept only `--config` as a named argument, then treat any additional `--name value` pairs as placeholder replacements. This is what lets one example YAML file run on different data files without editing the file each time.

```yaml
path: "<<run_path>>"
data:
  geometry: cartesian
  boundaries:
    - type: fits
      load_map: false
      files:
        Br: "<<Br>>"
        Bt: "<<Bt>>"
        Bp: "<<Bp>>"
```

Run it with custom files:

```bash
nf2-extrapolate \
  --config examples/configs/cartesian/minimal_fits.yaml \
  --run_path ./runs/ar377 \
  --Br ./data/ar377/Br.fits \
  --Bt ./data/ar377/Bt.fits \
  --Bp ./data/ar377/Bp.fits
```

You can also write literal paths directly in the YAML instead of placeholders:

```yaml
path: ./runs/ar377
data:
  geometry: cartesian
  boundaries:
    - type: fits
      load_map: false
      files:
        Br: ./data/ar377/Br.fits
        Bt: ./data/ar377/Bt.fits
        Bp: ./data/ar377/Bp.fits
```

Use placeholders for reusable example configs and literal paths for one-off local configs.

## What NF2 Does With The YAML

When a run starts, NF2 performs these steps:

1. Reads the YAML file.
2. Replaces `<<placeholder>>` values with matching command-line arguments.
3. Normalizes the public YAML schema into the internal runtime configuration.
4. Builds boundary, sampler, and validation datasets from `data`.
5. Builds the SIREN field model and configured losses.
6. Trains the model and writes `path/extrapolation_result.nf2`.

The `.nf2` result stores the trained model state and the normalized metadata needed by output helpers, exporters, and metrics.

## End-To-End Command Recipes

The examples below can be run from the repository root after installing NF2.
They use the same configuration files as `examples/scripts/`, but keep the full workflow in one place: prepare data, run an extrapolation, then inspect or export the result.

### Quick Smoke Test With Analytical Data

Analytical benchmark runs generate the boundary internally, so they are the fastest way to verify that the command-line tools, training loop, callbacks, and output loading work.

```bash
nf2-extrapolate \
  --config examples/configs/benchmark/analytical_case1.yaml \
  --run_path ./runs/benchmark/case1 \
  --work_path ./runs/benchmark/case1/work

nf2-metrics ./runs/benchmark/case1/extrapolation_result.nf2 \
  --Mm_per_pixel 0.05 \
  --height_range 0 2
```

Use `examples/configs/benchmark/analytical_case2.yaml` for a second benchmark geometry.

### SHARP CEA Active-Region Extrapolation

Download one HMI SHARP CEA vector magnetogram from JSOC, then run the Cartesian SHARP config.
Replace the email address with the email registered with JSOC.
Use `--noaa_num` instead of `--sharp_num` if you want NF2 to resolve the SHARP/HARP number from a NOAA active-region number.

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir ./data/sharp_cea_377 \
  --email you@example.org \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --series sharp_cea_720s \
  --segments Br,Bt,Bp,Br_err,Bt_err,Bp_err

nf2-extrapolate \
  --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/sharp_cea_377 \
  --work_path ./runs/sharp_cea_377/work \
  --Br ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits \
  --Bt ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits \
  --Bp ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits \
  --Br_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits \
  --Bt_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits \
  --Bp_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits
```

The output checkpoint is written to `./runs/sharp_cea_377/extrapolation_result.nf2`.

### Plain Br/Bt/Bp FITS Files

Use `minimal_fits.yaml` when the input files are plain Cartesian FITS arrays rather than SHARP CEA maps with SunPy/WCS metadata.

```bash
nf2-extrapolate \
  --config examples/configs/cartesian/minimal_fits.yaml \
  --run_path ./runs/cartesian_minimal \
  --work_path ./runs/cartesian_minimal/work \
  --Br ./data/plain_fits/Br.fits \
  --Bt ./data/plain_fits/Bt.fits \
  --Bp ./data/plain_fits/Bp.fits
```

This path is useful for preprocessed active-region cutouts or synthetic vector magnetograms that already contain Br, Bt, and Bp arrays on a Cartesian grid.

### Spherical Full-Disk HMI Extrapolation

Spherical runs combine a full-disk vector observation with Carrington synoptic maps.
The full-disk download converts native HMI vector data to spherical Br/Bt/Bp components through JSOC processing.

```bash
nf2-download \
  --source hmi_full_disk \
  --download_dir ./data/hmi_spherical/full_disk \
  --email you@example.org \
  --t_start 2011-02-15T00:00:00 \
  --series B_720s

nf2-download \
  --source hmi_synoptic \
  --download_dir ./data/hmi_spherical/synoptic \
  --email you@example.org \
  --carrington_rotation 2106 \
  --series b_synoptic \
  --segments Br,Bt,Bp

nf2-extrapolate \
  --config examples/configs/spherical/hmi_full_disk.yaml \
  --run_path ./runs/spherical_hmi \
  --work_path ./runs/spherical_hmi/work \
  --wandb_project nf2 \
  --run_name "Spherical HMI" \
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

For a quick example, the full-disk error placeholders above reuse the field maps.
For production runs, provide uncertainty maps that match your preprocessing.

### SHARP CEA Time Series

Series runs start from a completed single extrapolation.
First download a time sequence, train the first time step, then pass its `last.ckpt` to `nf2-extrapolate-series`.

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir ./data/sharp_cea_377 \
  --email you@example.org \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --t_end 2011-02-16T00:00:00 \
  --cadence 720s \
  --series sharp_cea_720s \
  --segments Br,Bt,Bp,Br_err,Bt_err,Bp_err

nf2-extrapolate \
  --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/sharp_cea_377_initial \
  --work_path ./runs/sharp_cea_377_initial/work \
  --Br ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits \
  --Bt ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits \
  --Bp ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits \
  --Br_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits \
  --Bt_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits \
  --Bp_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits

nf2-extrapolate-series \
  --config examples/configs/cartesian/sharp_cea_series.yaml \
  --run_path ./runs/sharp_cea_377_series \
  --work_path ./runs/sharp_cea_377_series/work \
  --meta_path ./runs/sharp_cea_377_initial/last.ckpt \
  --Br_pattern "./data/sharp_cea_377/*.Br.fits" \
  --Bt_pattern "./data/sharp_cea_377/*.Bt.fits" \
  --Bp_pattern "./data/sharp_cea_377/*.Bp.fits" \
  --Br_err_pattern "./data/sharp_cea_377/*.Br_err.fits" \
  --Bt_err_pattern "./data/sharp_cea_377/*.Bt_err.fits" \
  --Bp_err_pattern "./data/sharp_cea_377/*.Bp_err.fits"
```

Before starting a series, check that each glob matches the same number of files and that lexical sorting is chronological.
For multi-height series and list-based pairing, see [Series runs](series.md).

### Export And Quality Checks

After training, export fields for analysis or visualization and compute standard NLFF quality metrics.

```bash
nf2-export ./runs/sharp_cea_377/extrapolation_result.nf2 \
  --format vtk \
  --out ./runs/sharp_cea_377/exports/field.vtk \
  --Mm_per_pixel 1.44 \
  --height_range 0 80 \
  --metrics j alpha free_energy

nf2-metrics ./runs/sharp_cea_377/extrapolation_result.nf2 \
  --Mm_per_pixel 1.44 \
  --height_range 0 80
```

Use `--format npz` for Python workflows, `--format hdf5` for portable scientific output, and `--format vtk` for ParaView.

## Python API

The command-line tools are thin wrappers around the Python API. Use `nf2.run(...)` when you want to create configurations programmatically:

```python
import nf2

nf2.run(
    path="./runs/case1",
    data={
        "geometry": "cartesian",
        "boundaries": [
            {
                "type": "fits",
                "load_map": False,
                "files": {
                    "Br": "./data/Br.fits",
                    "Bt": "./data/Bt.fits",
                    "Bp": "./data/Bp.fits",
                },
            }
        ],
        "z_range": [0, 80],
    },
    training={"epochs": 30},
)
```

After training, load the result with `nf2.load(...)`. NF2 chooses `CartesianOutput` or `SphericalOutput` from the checkpoint metadata:

```python
import nf2

out = nf2.load("./runs/case1/extrapolation_result.nf2")
cube = out.load_cube(height_range=[0, 80], Mm_per_pixel=1.0, metrics=["j", "alpha"])
```

Use `nf2.run_series(...)` for time series, `nf2.export_file(...)` for programmatic exports, and `from nf2.metrics import compute_metrics` for custom evaluation procedures.

```{toctree}
:maxdepth: 2
:caption: Extrapolations

cartesian
spherical
analytical
series
```
