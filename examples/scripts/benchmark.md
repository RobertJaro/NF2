# Benchmark Analytical Cases

The analytical benchmark examples generate Low & Lou magnetic fields internally, so no input data download is required. They are useful as smoke tests after installation and as small runs for checking training, callbacks, exports, and metrics.

The benchmark configs intentionally keep the run compact:

- `training.epochs: 15`
- `data.iterations: 1000`
- no potential boundary
- no height loss scaling
- boundary validation plus height-slice validation

## Case 1

```bash
nf2-extrapolate \
  --config "examples/configs/benchmark/analytical_case1.yaml" \
  --run_path "./runs/benchmark/case1" \
  --work_path "./runs/benchmark/case1/work"
```

## Case 2

```bash
nf2-extrapolate \
  --config "examples/configs/benchmark/analytical_case2.yaml" \
  --run_path "./runs/benchmark/case2" \
  --work_path "./runs/benchmark/case2/work"
```

## After Training

The output is written to:

```text
runs/benchmark/case1/extrapolation_result.nf2
runs/benchmark/case2/extrapolation_result.nf2
```

Run metrics or exports with the commands in [export_metrics.md](export_metrics.md). For a quick check, use a small Cartesian export height range:

```bash
nf2-metrics "./runs/benchmark/case1/extrapolation_result.nf2" \
  --Mm_per_pixel 0.05 \
  --height_range 0 2
```
