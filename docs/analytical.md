# Analytical NLFF Cases

NF2 includes Low & Lou analytical magnetic fields for benchmark and regression use. Analytical cases use the same YAML-driven training path as observational Cartesian and spherical runs, but the boundary data are generated from a configured analytical field instead of read from FITS files.

Primary bundled config names:

- `nf2/benchmark/analytical_case1.yaml`
- `nf2/benchmark/analytical_case2.yaml`

These examples are intended for fast smoke tests, documentation, and quantitative validation of the training and export pipeline.

## Benchmark Analytical Commands

The analytical benchmark examples generate Low & Lou magnetic fields internally, so no input data download is required. They are useful as smoke tests after installation and as small runs for checking training, callbacks, exports, and metrics.

The benchmark configs intentionally keep the run compact:

- `training.epochs: 15`
- `data.iterations: 1000`
- no potential boundary
- no height loss scaling
- boundary validation plus height-slice validation

### Case 1

```bash
nf2-extrapolate \
  --config "nf2/benchmark/analytical_case1.yaml" \
  --run_path "./runs/benchmark/case1" \
  --work_path "./runs/benchmark/case1/work"
```

### Case 2

```bash
nf2-extrapolate \
  --config "nf2/benchmark/analytical_case2.yaml" \
  --run_path "./runs/benchmark/case2" \
  --work_path "./runs/benchmark/case2/work"
```

### After Training

The output is written to:

```text
runs/benchmark/case1/extrapolation_result.nf2
runs/benchmark/case2/extrapolation_result.nf2
```

Run metrics or exports with the commands in [Exporting](exporting.md) and [Quality metrics](metrics.md). For a quick check, use a small Cartesian export height range:

```bash
nf2-metrics "./runs/benchmark/case1/extrapolation_result.nf2" \
  --Mm_per_pixel 0.05 \
  --height_range 0 2
```

## How The YAML Is Configured

An analytical config is a Cartesian config with an `analytical` boundary:

```yaml
path: "<<run_path>>"
work_path: "<<work_path>>"
data:
  geometry: cartesian
  normalization:
    Mm_per_ds: 1
    Gauss_per_dB: 1
  boundaries:
    - id: boundary
      type: analytical
      case: 1
      boundary: full
      resolution: 64
      batch_size: 512
  sampler:
    type: height
    batch_size: 1024
  potential_boundary:
    type: none
  validation:
    - id: boundary
      type: analytical
      case: 1
      boundary: bottom
      resolution: 64
      batch_size: 512
    - id: slices
      type: slices
      n_slices: 5
      batch_size: 1024
  iterations: 1000
  z_range: [0, 2]
  validation_pixel_per_ds: 32
```

Run the example by filling the path placeholders:

```bash
nf2-extrapolate \
  --config nf2/benchmark/analytical_case1.yaml \
  --run_path ./runs/analytical_case1 \
  --work_path ./runs/analytical_case1/work
```

Use `case: 1` or `case: 2` to switch benchmark fields. Increase `resolution` for a denser boundary and validation grid, or reduce it for a faster smoke test. The default analytical bounds are `[-1, 1]` in `x`, `[-1, 1]` in `y`, and `[0, 2]` in `z`; with `Mm_per_ds: 1`, these coordinates are reported directly in Mm and are centered at `(0, 0)` in the boundary plane. The generated analytical field is normalized by the maximum absolute component value in the analytical cube, so with `Gauss_per_dB: 1` the training and validation values are approximately in the range `[-1, 1]` G.

## Losses And Validation

Analytical examples usually disable the potential boundary and use a compact set of losses:

```yaml
losses:
  - type: boundary
    name: boundary
    weight: 1.0
    datasets: boundary
  - type: force_free
    name: force_free
    weight: 1.0e-3
    datasets: [random]
loss_scaling: []
callbacks:
  - type: boundary
    dataset: boundary
  - type: slices
    dataset: slices
```

`loss_scaling: []` disables the default Cartesian height scaling for these benchmarks, so the force-free loss is used directly. The boundary callback compares the predicted and true lower boundary on physical coordinates. The slices callback logs magnetic-field height slices, current-density height slices, and vertically integrated current density.

## Python API

You can launch the same benchmark without a YAML file:

```python
import nf2

nf2.run(
    path="./runs/analytical_case1",
    data={
        "geometry": "cartesian",
        "normalization": {"Mm_per_ds": 1, "Gauss_per_dB": 1},
        "boundaries": [
            {
                "id": "boundary",
                "type": "analytical",
                "case": 1,
                "boundary": "full",
                "resolution": 64,
                "batch_size": 512,
            }
        ],
        "sampler": {"type": "height", "batch_size": 1024},
        "potential_boundary": {"type": "none"},
        "validation": [
            {
                "id": "boundary",
                "type": "analytical",
                "case": 1,
                "boundary": "bottom",
                "resolution": 64,
                "batch_size": 512,
            },
            {"id": "slices", "type": "slices", "n_slices": 5, "batch_size": 1024},
        ],
        "z_range": [0, 2],
        "validation_pixel_per_ds": 32,
    },
    training={"epochs": 15},
    loss_scaling=[],
    callbacks=[
        {"type": "boundary", "dataset": "boundary"},
        {"type": "slices", "dataset": "slices"},
    ],
)
```

Load the result like any other Cartesian run:

```python
import nf2

out = nf2.load("./runs/analytical_case1/extrapolation_result.nf2")
cube = out.load_cube(height_range=[0, 2], Mm_per_pixel=0.02, metrics=["j"])
```
