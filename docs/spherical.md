# Spherical Extrapolations

Spherical runs use `data.geometry: spherical` and train a global or large-field-of-view volume in spherical coordinates. They are intended for full-disk vector maps, synoptic maps, and combined full-disk plus synoptic constraints.

The full-disk plus synoptic example configuration is:

- `examples/configs/spherical/hmi_full_disk.yaml`

## How The YAML Is Configured

Spherical YAML files set `data.geometry: spherical`, define an outer radius, then list map boundaries:

```yaml
path: "<<run_path>>"
work_path: "<<work_path>>"
data:
  geometry: spherical
  max_radius: 1.3
  boundaries:
    - id: full_disk
      type: map
      files:
        Br: "<<full_disk_Br>>"
        Bt: "<<full_disk_Bt>>"
        Bp: "<<full_disk_Bp>>"
    - id: synoptic
      type: map
      files:
        Br: "<<synoptic_Br>>"
        Bt: "<<synoptic_Bt>>"
        Bp: "<<synoptic_Bp>>"
```

Fill the placeholders from the command line:

```bash
nf2-extrapolate \
  --config examples/configs/spherical/hmi_full_disk.yaml \
  --run_path ./runs/hmi_spherical \
  --work_path ./runs/hmi_spherical/work \
  --full_disk_Br ./data/full_disk/Br.fits \
  --full_disk_Bt ./data/full_disk/Bt.fits \
  --full_disk_Bp ./data/full_disk/Bp.fits \
  --full_disk_Br_err ./data/full_disk/Br_err.fits \
  --full_disk_Bt_err ./data/full_disk/Bt_err.fits \
  --full_disk_Bp_err ./data/full_disk/Bp_err.fits \
  --synoptic_Br ./data/synoptic/Br.fits \
  --synoptic_Bt ./data/synoptic/Bt.fits \
  --synoptic_Bp ./data/synoptic/Bp.fits
```

For a fixed local config, write the file paths directly in the YAML:

```yaml
data:
  geometry: spherical
  max_radius: 1.3
  boundaries:
    - id: full_disk
      type: map
      files:
        Br: ./data/full_disk/Br.fits
        Bt: ./data/full_disk/Bt.fits
        Bp: ./data/full_disk/Bp.fits
```

## Spherical Single-Run Example

Spherical runs combine full-disk HMI vector data with synoptic maps and train in a spherical coordinate volume. The example config is `examples/configs/spherical/hmi_full_disk.yaml`.

### 1. Download Full-Disk HMI Data

This command downloads a full-disk vector field and converts it to spherical Br/Bt/Bp components through JSOC's `HmiB2ptr` processing step.

```bash
nf2-download \
  --source hmi_full_disk \
  --download_dir "./data/hmi_spherical/full_disk" \
  --email "you@example.org" \
  --t_start 2011-02-15T00:00:00 \
  --series B_720s
```

### 2. Download Carrington Synoptic Maps

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --t_start "2011-02-15T00:00:00" \
  --series b_synoptic \
  --segments Br,Bt,Bp
```

You can also pass `--carrington_rotation` directly. Use `--synoptic_product mr_polfil` to download files such as `hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits`.

### 3. Run The Spherical Config

Fill the placeholders with the downloaded files. The full-disk error placeholders can point to dedicated uncertainty maps; if those are not available, use the same field maps for a quick example and prepare proper uncertainties for production runs.

```bash
nf2-extrapolate \
  --config "examples/configs/spherical/hmi_full_disk.yaml" \
  --run_path "./runs/spherical_hmi" \
  --work_path "./runs/spherical_hmi/work" \
  --wandb_project "nf2" \
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

### 4. Memory Notes

Spherical runs are usually heavier than compact Cartesian cutouts. Reduce `batch_size`, `n_lat_lon_sample`, validation sampling, or radial/latitude/longitude ranges if you hit out-of-memory errors.

## Boundary Data

The `map` dataset reads spherical `Br`, `Bt`, and `Bp` maps. Error maps can be supplied under `errors`; NF2 merges them into the dataset internally:

```yaml
boundaries:
  - id: full_disk
    type: map
    batch_size: 8192
    requires_jacobian: false
    files:
      Br: "<<full_disk_Br>>"
      Bt: "<<full_disk_Bt>>"
      Bp: "<<full_disk_Bp>>"
    errors:
      Br_err: "<<full_disk_Br_err>>"
      Bt_err: "<<full_disk_Bt_err>>"
      Bp_err: "<<full_disk_Bp_err>>"
    mask_configs:
      type: mu_filter
      min: 0.2
```

The `mask_configs` block is commonly used to suppress low-confidence limb pixels in full-disk observations. Synoptic maps usually omit the error files and mask.

## Samplers, Validation, And Losses

If `data.samplers` is omitted, NF2 adds a `random_radial_grouped` sampler. For explicit control:

```yaml
data:
  iterations: 10000
  samplers:
    - id: random
      type: random_radial_grouped
      batch_size: 16384
      n_lat_lon_sample: 64
      radial_sampling_exponent: 2
  validation:
    - id: sphere
      type: sphere
      resolution: 128
    - id: slices
      type: spherical_slices
      longitude_resolution: 128
      n_slices: 5
```

Spherical losses usually combine boundary matching with volume regularization:

```yaml
losses:
  - type: boundary
    name: boundary
    weight: 1.0
    datasets: [full_disk, synoptic]
  - type: force_free
    name: force_free
    weight: { start: 1.0e-4, end: 1.0e-2, iterations: 50000 }
    datasets: [random]
  - type: potential
    name: potential
    weight: { start: 1.0e-4, end: 1.0e-2, iterations: 50000 }
    datasets: [random]
```

Use `loss_scaling.type: radial` to scale selected volume losses across radius.

## Python API

Use `nf2.run(...)` with `geometry: spherical` for programmatic runs:

```python
import nf2

nf2.run(
    path="./runs/hmi_spherical",
    data={
        "geometry": "spherical",
        "max_radius": 1.3,
        "boundaries": [
            {
                "id": "full_disk",
                "type": "map",
                "files": {
                    "Br": "./data/full_disk/Br.fits",
                    "Bt": "./data/full_disk/Bt.fits",
                    "Bp": "./data/full_disk/Bp.fits",
                },
            },
            {
                "id": "synoptic",
                "type": "map",
                "files": {
                    "Br": "./data/synoptic/Br.fits",
                    "Bt": "./data/synoptic/Bt.fits",
                    "Bp": "./data/synoptic/Bp.fits",
                },
            },
        ],
    },
    training={"epochs": 100},
)
```

The primary output helper is selected automatically by `nf2.load(...)`:

```python
from astropy import units as u
import nf2

out = nf2.load("./runs/hmi_spherical/extrapolation_result.nf2")
volume = out.load_spherical(
    radius_range=[1.0, 1.3] * u.solRad,
    sampling=[100, 180, 360],
    metrics=["j", "alpha"],
)
```
