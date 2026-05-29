# Configuration

NF2 uses a single public YAML schema for Cartesian, spherical, and benchmark runs. The loader expands this public schema into the explicit runtime configuration used by the training modules.

See the generated [full YAML reference](generated/configuration_reference.md) and [dataset reference](generated/datasets_reference.md) for the complete option list.

## Minimal Cartesian Example

```yaml
path: ./runs/sharp_cea
data:
  geometry: cartesian
  boundaries:
    - type: fits
      load_map: false
      files:
        Br: ./data/Br.fits
        Bt: ./data/Bt.fits
        Bp: ./data/Bp.fits
```

This uses defaults for SIREN model, random sampling, potential boundary data, losses, callbacks, normalization, and training settings.

## Minimal Spherical Example

```yaml
path: ./runs/spherical
data:
  geometry: spherical
  boundaries:
    - id: full_disk
      type: map
      files:
        Br: ./data/full_disk.Br.fits
        Bt: ./data/full_disk.Bt.fits
        Bp: ./data/full_disk.Bp.fits
```

## Top-Level Keys

- `path`: result directory. The trained model is written to `path/extrapolation_result.nf2`.
- `work_path`: optional scratch directory. If omitted, NF2 uses `path/work`.
- `logging`: passed to the Lightning W&B logger.
- `data`: geometry, boundaries, samplers, validation data, and normalization.
- `model`: optional SIREN field model configuration.
- `training`: optimizer, validation cadence, and Lightning trainer settings.
- `losses`: optional explicit losses. Use `weight`; `lambda` is not supported.
- `loss_scaling`: optional spatial loss scaling modules.
- `callbacks`: optional validation plots and metrics.

## Placeholder Overrides

Any `<<name>>` placeholder in YAML can be filled from the command line:

```yaml
path: "<<run_path>>"
data:
  geometry: cartesian
  boundaries:
    - type: sharp
      files:
        Br: "<<Br>>"
        Bt: "<<Bt>>"
        Bp: "<<Bp>>"
```

Run with:

```bash
nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/ar377 \
  --work_path /scratch/ar377 \
  --Br /data/Br.fits \
  --Bt /data/Bt.fits \
  --Bp /data/Bp.fits
```

This is useful when the same example config should run on different file systems.

Spherical series configs use a separate `[[dataset.path.to.value]]` notation for values resolved during dataset loading. For example, `[[full_disk.files.Br]]` points to the current expanded full-disk `Br` file and is not filled from the command line.

## Data

`data.geometry` is required and must be `cartesian` or `spherical`.

Common data keys:

- `normalization.Mm_per_ds`: model length unit in Mm.
- `normalization.Gauss_per_dB`: model magnetic-field unit in Gauss.
- `boundaries`: observed or analytical boundary datasets.
- `sampler` or `samplers`: physics sampling datasets.
- `validation`: validation and plotting datasets.

If omitted, `normalization.Mm_per_ds` defaults to `100` and `normalization.Gauss_per_dB` defaults to `1000`.

Cartesian-specific keys:

- `z_range`: extrapolation height range in Mm.
- `potential_boundary`: explicit potential boundary data. FFT potential fields are used by default; set `method: direct` to use the Green's-function fallback. Use `{type: none}` to disable.
- `iterations`: number of random sampler batches per epoch-like pass.

Spherical-specific keys:

- `max_radius`: outer radius in solar radii.
- `samplers`: spherical physics samplers, usually `random_radial_grouped`. If omitted, NF2 adds a default random radial sampler.

## Boundaries And Files

Every boundary should have an `id` when more than one dataset is used. Losses, callbacks, and transforms refer to these ids through `datasets` or `dataset`.

Cartesian boundaries can set their own spatial scale and placement:

```yaml
data:
  geometry: cartesian
  boundaries:
    - id: hmi
      type: sharp
      Mm_per_pixel: 0.36
      coordinate_center: [0, 0]
      files:
        Br: ./hmi.Br.fits
        Bt: ./hmi.Bt.fits
        Bp: ./hmi.Bp.fits
    - id: chromosphere
      type: los_trv_azi
      Mm_per_pixel: 0.09
      coordinate_center:
        x: 12.0
        y: -4.5
      files:
        B_los: ./chromosphere.B_los.fits
        B_trv: ./chromosphere.B_trv.fits
        B_azi: ./chromosphere.B_azi.fits
```

By default, NF2 places the center of each Cartesian boundary at `[0, 0]` Mm. Set `coordinate_center` to place the center of an instrument field of view at a defined model coordinate. This lets multiple instruments use independent `Mm_per_pixel` values while training in one shared Cartesian domain.

Common Cartesian FITS layouts use:

```yaml
files:
  Br: ./Br.fits
  Bt: ./Bt.fits
  Bp: ./Bp.fits
```

LOS/transverse/azimuth inputs use the same public file-map style, but the loader converts them into vector components and can perform automatic disambiguation when configured by the example:

```yaml
files:
  B_los: ./B_los.fits
  B_trv: ./B_trv.fits
  B_azi: ./B_azi.fits
```

Cartesian and spherical vector boundaries use `Br`, `Bt`, and `Bp` files. Error maps can be placed under `errors`; NF2 routes them to the loader-specific internal form.

## Potential Boundary

Cartesian runs add an explicit potential boundary by default:

```yaml
potential_boundary:
  type: potential
  strides: 4
  method: fft
```

Use `type: none` to disable it. Prefer the FFT method for potential fields and free-energy estimates; use `method: direct` only when a direct Green's-function comparison is needed.

## Model

All trainable models use SIREN networks. If omitted, NF2 uses a vector-potential SIREN.

```yaml
model:
  field: vector_potential
  network:
    type: siren
    hidden_dim: 256
    layers: 8
    w0: 1.0
    w0_initial: 5.0
```

Supported `field` values:

- `vector_potential`: predicts `A` and derives `B = curl(A)`.
- `b`: predicts `B` directly.

## Training

The minimal default trains for 10 epochs with an exponential Adam learning-rate schedule. All important trainer settings can still be set explicitly:

```yaml
training:
  epochs: 30
  validation_interval: 1000
  check_val_every_n_epoch: 1
  gradient_clip_val: 0.1
  matmul_precision: medium
  optimizer:
    start: 5.0e-4
    end: 5.0e-5
    iterations: 100000
  trainer:
    accelerator: gpu
    devices: 1
    precision: 32
    num_sanity_val_steps: 0
```

Entries under `training.trainer` are passed to the Lightning `Trainer` after NF2 sets its defaults.

## Losses

Losses use `weight`, not `lambda`.

```yaml
losses:
  - type: boundary
    name: boundary
    weight: 1.0
    datasets: [boundary]
  - type: force_free
    name: force_free
    weight: 1.0e-4
    datasets: [random]
```

Scheduled weights are also supported:

```yaml
weight:
  type: step
  steps: 5000
  start: 1.0e-4
  end: 0.0
```

## Example Layout

- `examples/configs/cartesian`: observational Cartesian configs.
- `examples/configs/spherical`: spherical full-disk/synoptic configs.
- `examples/configs/benchmark`: analytical benchmark configs.
