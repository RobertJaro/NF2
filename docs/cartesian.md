# Cartesian Extrapolations

Cartesian runs use `data.geometry: cartesian`.

Primary examples:

- `examples/configs/cartesian/minimal_fits.yaml`
- `examples/configs/cartesian/sharp_cea.yaml`
- `examples/configs/cartesian/auto_disambiguation.yaml`
- `examples/configs/cartesian/multi_height.yaml`
- `examples/configs/cartesian/multi_height_series.yaml`

The primary output helper is:

```python
from nf2 import CartesianOutput

out = CartesianOutput("extrapolation_result.nf2")
cube = out.load_cube(height_range=[0, 80], Mm_per_pixel=0.72, metrics=["j"])
slice0 = out.load_slice()
```

## Minimal Configuration

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

This keeps the default SIREN model, normalization, random sampler, potential boundary, losses, callbacks, and training options.

## Multi-Height Data

Multi-height configurations use multiple boundary entries with distinct ids. They can also be used in series runs when each component pattern expands to the same number of files.

Each boundary can define its own `Mm_per_pixel` and `coordinate_center`. The default center is `[0, 0]` Mm, matching the previous centered-coordinate behavior. When several boundaries are provided, NF2 builds the Cartesian training domain from the union of their XY coordinate ranges, so offset instruments can be combined consistently.
