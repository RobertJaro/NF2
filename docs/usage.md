# Usage Overview

NF2 can be used from YAML-driven commands or from Python.

## Command Line

Use `nf2-extrapolate` for one run:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea" \
  --Br "./data/Br.fits" \
  --Bt "./data/Bt.fits" \
  --Bp "./data/Bp.fits"
```

Use `nf2-extrapolate-series` when file placeholders are glob patterns:

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --photosphere_B_los_pattern "./data/photosphere/*B_los.fits"
```

Use `nf2-export` and `nf2-metrics` after training:

```bash
nf2-export "./runs/sharp_cea/extrapolation_result.nf2" --format vtk --metrics j alpha free_energy_fft
nf2-metrics "./runs/sharp_cea/extrapolation_result.nf2" --Mm_per_pixel 1.44 --height_range 0 80
```

The generated [CLI reference](generated/cli_reference.md) lists all command options.

## Python

```python
import nf2

nf2.run(
    path="./runs/case1",
    data={"geometry": "cartesian", "boundaries": [{"type": "analytical", "case": 1}]},
)

out = nf2.load("./runs/case1/extrapolation_result.nf2")
cube = out.load_cube(Mm_per_pixel=1.0, height_range=[0, 80], metrics=["j"])
```
