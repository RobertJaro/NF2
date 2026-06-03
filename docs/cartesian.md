# Cartesian Extrapolations

Cartesian runs use `data.geometry: cartesian` and train a local Cartesian volume above one or more planar boundary maps. This is the usual choice for active-region cutouts, SHARP CEA data, generic FITS vector maps, LOS/transverse/azimuth inputs, and multi-height observations.

Primary bundled config names:

- `nf2/cartesian/minimal_fits.yaml`
- `nf2/cartesian/sharp_cea.yaml`
- `nf2/cartesian/auto_disambiguation.yaml`
- `nf2/cartesian/multi_height.yaml`
- `nf2/cartesian/sharp_cea_series.yaml`
- `nf2/cartesian/multi_height_series.yaml`

## Cartesian Single-Run Examples

Cartesian runs train a local volume above one or more planar boundary maps. Use these examples for SHARP CEA data, generic plain FITS arrays, and multi-height observations.

### SHARP CEA Data

First download a SHARP CEA vector magnetogram from JSOC. Fill in your JSOC email and either a SHARP number or NOAA number.

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

Then run the SHARP CEA config. The `sharp` dataset reads SHARP CEA maps with map metadata. Error-map arguments are optional; omit `--Br_err`, `--Bt_err`, and `--Bp_err` if those files are not available.

```bash
nf2-extrapolate \
  --config "nf2/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea_377" \
  --work_path "./runs/sharp_cea_377/work" \
  --Br "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits" \
  --Bt "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits" \
  --Bp "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits" \
  --Br_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits" \
  --Bt_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits" \
  --Bp_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
```

### Plain FITS Arrays

Use `minimal_fits.yaml` for plain Br/Bt/Bp FITS data arrays without SunPy map metadata. This config uses `type: fits` with `load_map: false`.

```bash
nf2-extrapolate \
  --config "nf2/cartesian/minimal_fits.yaml" \
  --run_path "./runs/cartesian_minimal" \
  --Br "./data/plain_fits/Br.fits" \
  --Bt "./data/plain_fits/Bt.fits" \
  --Bp "./data/plain_fits/Bp.fits"
```

Use this path when your FITS files only contain array data and do not need coordinate information from FITS/WCS headers.

### Multi-Height Cartesian Data

Multi-height examples assume you already prepared matching photospheric and chromospheric FITS files. No generic download command is provided because the source instruments and preprocessing are project-specific.

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

Keep this first multi-height run if you plan to start a series. Its `last.ckpt` becomes the `meta_path` for the series run.

### Common Adjustments

- Reduce `data.sampler.batch_size`, boundary `batch_size`, or validation `batch_size` if training runs out of GPU memory.
- Use `WANDB_MODE=offline` or `WANDB_MODE=disabled` if you do not want online W&B logging.
- Run `nf2-metrics` after training to check boundary quality, divergence, current alignment, and magnetic energy.

## How The YAML Is Configured

The minimal Cartesian YAML needs a run path, the Cartesian geometry, and a boundary dataset:

```yaml
path: "<<run_path>>"
data:
  geometry: cartesian
  boundaries:
    - id: boundary
      type: fits
      load_map: false
      files:
        Br: "<<Br>>"
        Bt: "<<Bt>>"
        Bp: "<<Bp>>"
```

The placeholders are filled from the command line:

```bash
nf2-extrapolate \
  --config nf2/cartesian/minimal_fits.yaml \
  --run_path ./runs/ar377 \
  --Br ./data/ar377/Br.fits \
  --Bt ./data/ar377/Bt.fits \
  --Bp ./data/ar377/Bp.fits
```

For a project-specific config, replace the placeholders with literal file paths:

```yaml
path: ./runs/ar377
data:
  geometry: cartesian
  boundaries:
    - id: boundary
      type: fits
      load_map: false
      files:
        Br: ./data/ar377/Br.fits
        Bt: ./data/ar377/Bt.fits
        Bp: ./data/ar377/Bp.fits
```

The `type` selects how NF2 reads the data. Common Cartesian boundary types are:

- `sharp`: SHARP CEA `Br`, `Bt`, and `Bp` FITS files.
- `fits`: generic Cartesian `Br`, `Bt`, and `Bp` FITS files.
- `los_trv_azi`: line-of-sight, transverse-field, and azimuth FITS files.
- `los`: line-of-sight-only boundary data.
- `analytical`: generated benchmark fields.

## Domain, Sampling, And Validation

Cartesian configs commonly set `z_range`, a random volume sampler, a potential boundary, and validation datasets:

```yaml
data:
  geometry: cartesian
  z_range: [0, 80]
  sampler:
    type: height
    batch_size: 16384
  potential_boundary:
    type: potential
    strides: 4
  validation:
    - id: boundary
      type: sharp
      files:
        Br: "<<Br>>"
        Bt: "<<Bt>>"
        Bp: "<<Bp>>"
    - id: cube
      type: cube
    - id: slices
      type: slices
```

`z_range` is measured in Mm. The random sampler provides volume points for force-free and divergence-related losses. The potential boundary adds side/top boundary constraints by default; set `type: none` to disable it. Validation datasets drive callbacks and metrics during training.

## Losses

Explicit losses make the training objective clear:

```yaml
losses:
  - type: boundary
    name: boundary
    weight: 1.0
    datasets: [boundary]
  - type: force_free
    name: force_free
    weight: 1.0e-3
    datasets: [random]
  - type: potential
    name: potential
    weight: { type: step, steps: 5000, start: 1.0e-4, end: 0.0 }
    datasets: [random]
```

The `datasets` entries refer to dataset ids from `data.boundaries`, `data.sampler`, or generated defaults such as `random` and `potential`.

## Python API

Use `nf2.run(...)` for programmatic Cartesian runs:

```python
import nf2

nf2.run(
    path="./runs/ar377",
    data={
        "geometry": "cartesian",
        "boundaries": [
            {
                "id": "boundary",
                "type": "fits",
                "load_map": False,
                "files": {
                    "Br": "./data/ar377/Br.fits",
                    "Bt": "./data/ar377/Bt.fits",
                    "Bp": "./data/ar377/Bp.fits",
                },
            }
        ],
        "z_range": [0, 80],
    },
    training={"epochs": 30},
)
```

The primary output helper for trained Cartesian results is:

```python
import nf2

out = nf2.load("./runs/ar377/extrapolation_result.nf2")
cube = out.load_cube(height_range=[0, 80], Mm_per_pixel=0.72, metrics=["j"])
slice0 = out.load_slice()
```

## Multi-Height Data

Multi-height configurations use multiple boundary entries with distinct ids. Each boundary can point to its own custom files and define its own plate scale and placement:

```yaml
data:
  geometry: cartesian
  boundaries:
    - id: photosphere
      type: los_trv_azi
      Mm_per_pixel: 0.09
      files:
        B_los: "<<photosphere_B_los>>"
        B_trv: "<<photosphere_B_trv>>"
        B_azi: "<<photosphere_B_azi>>"
      ambiguous_azimuth: true
      load_map: false
    - id: chromosphere
      type: los_trv_azi
      Mm_per_pixel: 0.09
      files:
        B_los: "<<chromosphere_B_los>>"
        B_trv: "<<chromosphere_B_trv>>"
        B_azi: "<<chromosphere_B_azi>>"
      height_mapping: { z: 2.0, z_min: 0.0, z_max: 20.0 }
      ambiguous_azimuth: true
      load_map: false
```

Run this config by passing custom files for each placeholder:

```bash
nf2-extrapolate \
  --config nf2/cartesian/multi_height.yaml \
  --run_path ./runs/multi_height_initial \
  --work_path ./runs/multi_height_initial/work \
  --photosphere_B_los ./data/photosphere/B_los.fits \
  --photosphere_B_trv ./data/photosphere/B_trv.fits \
  --photosphere_B_azi ./data/photosphere/B_azi.fits \
  --chromosphere_B_los ./data/chromosphere/B_los.fits \
  --chromosphere_B_trv ./data/chromosphere/B_trv.fits \
  --chromosphere_B_azi ./data/chromosphere/B_azi.fits
```

Each boundary can define its own `Mm_per_pixel` and `coordinate_center`. The default center is `[0, 0]` Mm, matching the previous centered-coordinate behavior. When several boundaries are provided, NF2 builds the Cartesian training domain from the union of their XY coordinate ranges, so offset instruments can be combined consistently.

Multi-height configs can also be used in [series runs](series.md) when each component pattern expands to the same number of files.
