# Training

Training behavior is controlled by the `training`, `losses`, `loss_scaling`, `data.sampler`, and `data.validation` sections of the YAML config. The examples in `examples/configs` are the best starting point for complete files.

```{toctree}
:maxdepth: 1
:caption: Training Details

configuration
```

## Loss Setup

Losses are listed under `losses`. Every explicit loss needs a `type`, a stable `name`, a `weight`, and usually one or more dataset ids.

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

Use `datasets` to point a loss at boundary, sampler, or validation dataset ids. NF2 v0.4 uses `weight`; the legacy `lambda` key is rejected.

Common Cartesian losses include `boundary`, `force_free`, and `potential`. Multi-height LOS/transverse/azimuth configs often use `boundary_los_trv_azi` plus a `height` loss on the elevated boundary. Spherical configs usually combine `boundary`, `force_free`, `potential`, and sometimes `energy_gradient`.

## Loss Schedules

A loss `weight` can be a number or a schedule. Supported schedule types are `exponential`, `linear`, and `step`. If `type` is omitted, NF2 uses an exponential schedule.

```yaml
losses:
  - type: force_free
    name: force_free
    weight:
      type: exponential
      start: 1.0e-4
      end: 1.0e-2
      iterations: 50000
    datasets: [random]
  - type: potential
    name: potential
    weight:
      type: step
      steps: 5000
      start: 1.0e-4
      end: 0.0
    datasets: [random]
```

Use schedules when one objective should enter gradually or disappear after a warm-up. A common Cartesian pattern is to turn off the potential loss after the model has learned the initial large-scale structure.

## Height Scaling

Loss scaling changes how strongly selected losses contribute across height or radius. Cartesian examples use `b_height` scaling for volume losses:

```yaml
loss_scaling:
  - type: b_height
    name: b_height
    loss_ids: [force_free, potential]
```

Spherical examples use radial scaling:

```yaml
loss_scaling:
  - type: radial
    name: radial
    base_radius: 1.0
    max_radius: 1.3
    loss_ids: [force_free, potential, energy_gradient]
```

For multi-height data, set `height_mapping` on the elevated boundary and add a matching height transform:

```yaml
data:
  boundaries:
    - id: chromosphere
      type: los_trv_azi
      height_mapping: { z: 2.0, z_min: 0.0, z_max: 20.0 }
transforms:
  - type: height
    height_range: [0, 20]
    datasets: [chromosphere]
```

## Batch Sizes

Training memory is mostly controlled by dataset batch sizes. Start by reducing the largest sampler or boundary batches:

```yaml
data:
  sampler:
    type: height
    batch_size: 8192
  potential_boundary:
    type: potential
    strides: 4
    batch_size: 4096
  validation_batch_size: 4096
```

For spherical `random_radial_grouped` samplers, `batch_size` must be divisible by `n_lat_lon_sample`.

```yaml
data:
  samplers:
    - id: random
      type: random_radial_grouped
      batch_size: 8192
      n_lat_lon_sample: 64
```

Validation can use a smaller `batch_size` than training. This is useful when callbacks or metrics run out of memory even though training batches fit.

## Loader And Series Cadence

NF2 defaults to 4 PyTorch DataLoader workers. On shared filesystems or series runs with frequent DataLoader reloads, lowering validation workers often reduces transition overhead:

```yaml
data:
  num_workers: 4
  validation_num_workers: 0
  prefetch_factor: 2
```

Series configs advance to a new dataset every epoch by default. The example series configs validate every 10th dataset while still saving one `.nf2` result per dataset:

```yaml
training:
  reload_dataloaders_every_n_epochs: 1
  check_val_every_n_epoch: 10
```

If preloading every series step uses too much memory, set `data.preload_data_modules: false` to load only the active step.

## Validation Resolution

Large active regions can make Cartesian validation cubes expensive. Reduce validation resolution by increasing `ds_per_pixel` on `cube` or `slices`, or by lowering the global validation density.

```yaml
data:
  validation_pixel_per_ds: 64
  validation:
    - id: cube
      type: cube
      ds_per_pixel: 0.03125
      batch_size: 4096
    - id: slices
      type: slices
      n_slices: 6
      batch_size: 4096
```

For spherical validation, reduce `sphere.resolution`, `spherical_slices.longitude_resolution`, or `spherical_slices.n_slices`.

```yaml
data:
  validation:
    - id: sphere
      type: sphere
      resolution: 128
      batch_size: 1024
    - id: slices
      type: spherical_slices
      longitude_resolution: 128
      n_slices: 5
```

## Out Of Memory Errors

When training fails with CUDA out-of-memory errors, reduce memory in this order:

1. Lower `data.sampler.batch_size` or spherical sampler `batch_size`.
2. Lower boundary dataset `batch_size`, especially high-resolution full-disk maps.
3. Reduce validation `batch_size` and validation resolution.
4. Increase `potential_boundary.strides` for Cartesian runs.
5. Reduce `model.network.hidden_dim` only after data and validation batches have been tuned.

If the error happens only during export or `nf2-metrics`, pass a smaller evaluation batch size to the command:

```bash
nf2-metrics ./runs/case/extrapolation_result.nf2 --batch_size 2048 --Mm_per_pixel 1.44 --height_range 0 80
```
