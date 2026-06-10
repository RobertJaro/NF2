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

Defaults are meant to make a valid first run, not to encode every scientific choice. They are a good starting point for smoke tests, SHARP-like Cartesian examples, and quick checks that the data loader works. Review them before production runs, especially the physical normalization, domain size, loss weights, batch sizes, and validation resolution.

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
nf2-extrapolate --config nf2/cartesian/sharp_cea.yaml \
  --run_path ./runs/ar377 \
  --work_path /scratch/ar377 \
  --Br /data/Br.fits \
  --Bt /data/Bt.fits \
  --Bp /data/Bp.fits
```

This is useful when the same example config should run on different file systems.

Placeholders can include defaults with `<<name;default>>`. If the command-line argument is omitted, the default YAML fragment is used; if no default is present, the config loader raises an error. Defaults can be scalar values or complete YAML snippets:

```yaml
data:
  z_range: <<z_range;[0, 100]>>
losses:
  - type: force_free
    name: force_free
    weight: <<force_free_weight;1.0e-3>>
  - type: potential
    name: potential
    weight: { type: step, steps: 5000, start: <<force_free_weight;1.0e-3>>, end: 0.0 }
```

The bundled Cartesian configs expose these defaults, so `--z_range 0 200` sets the height range to `[0, 200]`, and `--force_free_weight 2.0e-3` updates both the force-free loss weight and the potential-loss starting weight.

Placeholder resolution happens before YAML parsing:

- `--name value` replaces every matching `<<name>>` or `<<name;default>>`.
- Multi-value arguments such as `--z_range 0 200` are rendered as YAML lists, for example `[0, 200]`.
- If the CLI argument is missing and the placeholder has a default, the default text is inserted as YAML.
- If the CLI argument is missing and the placeholder has no default, loading fails with a missing-argument error.

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

Normalization controls the scale seen by the neural network. Keep the defaults when using the packaged examples unless you have a reason to change the model units. Override `Mm_per_ds` or `Gauss_per_dB` when the physical size or field scale of your data differs strongly from the examples, or when reproducing a published configuration that specifies those values. Export commands and output helpers convert back to physical units using the normalization stored in the checkpoint.

Cartesian-specific keys:

- `z_range`: extrapolation height range in Mm.
- `potential_boundary`: explicit potential boundary data. FFT potential fields are used by default; set `method: direct` to use the Green's-function fallback. Use `{type: none}` to disable.
- `iterations`: number of random sampler batches per epoch-like pass.

The bundled Cartesian configs default to `z_range: [0, 100]` Mm through `<<z_range;[0, 100]>>`. Override it from the command line when the extrapolated height should differ, for example `--z_range 0 150`. Use a lower range for smoke tests and memory-limited checks, then increase it for the scientific volume. Keep the default potential boundary unless you are running analytical benchmarks, custom side-boundary experiments, or a config that deliberately disables it.

Spherical-specific keys:

- `max_radius`: outer radius in solar radii.
- `samplers`: spherical physics samplers, usually `random_radial_grouped`. If omitted, NF2 adds a default random radial sampler.

For spherical runs, `max_radius`, latitude/longitude ranges, and sampler sizes determine most of the memory cost. Start with the packaged spherical example, reduce `batch_size` or validation resolution if memory is tight, and only then change model width or depth.

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

Cartesian and spherical vector boundaries use `Br`, `Bt`, and `Bp` files. Error maps can be placed under `errors`; NF2 routes them to the loader-specific internal form. In bundled templates, unresolved error placeholders are optional and are skipped when the matching command-line arguments are omitted.

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
- `scaled_vector_potential`: predicts `A` from radially compressed coordinates, scales it by `(r / R_sun)^-p`, and derives `B = curl(A)`. The default `radial_power` is `2`, matching a dipole-like vector-potential falloff. The default `coordinate_radial_power` is `4`, so the SIREN input scale is about `0.35` at `1.3 R_sun`.
- `b`: predicts `B` directly.

## Training

The minimal default trains for 15 epochs with an exponential Adam learning-rate schedule. All important trainer settings can still be set explicitly:

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

The default `training.epochs: 15` is deliberately conservative. It is enough for quick functional checks, but observational extrapolations often need more epochs or more `data.iterations`. Use validation plots, loss curves, and `nf2-metrics` to decide whether a run has stabilized.

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
    weight: <<force_free_weight;1.0e-3>>
    datasets: [random]
```

Scheduled weights are also supported:

```yaml
weight:
  type: step
  steps: 5000
  start: <<force_free_weight;1.0e-3>>
  end: 0.0
```

Explicit losses are useful when you want the YAML to record the full scientific objective. If `losses` is omitted, NF2 inserts geometry-appropriate defaults. When tuning a run, change one loss weight or schedule at a time and compare metrics on the same exported volume.

## Bundled Config Names

Installed NF2 includes reusable YAML templates. Pass them to `--config` with the `nf2/...` prefix from any working directory:

- `nf2/cartesian`: observational Cartesian configs.
- `nf2/spherical`: spherical full-disk/synoptic configs.
- `nf2/benchmark`: analytical benchmark configs.
