# Spherical Extrapolations

Spherical runs use `data.geometry: spherical`.

The full-disk plus synoptic example configuration is:

- `examples/configs/spherical/full_disk_synoptic.yaml`

The primary output helper is:

```python
from astropy import units as u
from nf2 import SphericalOutput

out = SphericalOutput("extrapolation_result.nf2")
volume = out.load_spherical(
    radius_range=[1.0, 1.3] * u.solRad,
    sampling=[100, 180, 360],
    metrics=["j", "alpha"],
)
```

## Boundary Data

The spherical example combines full-disk HMI vector data with synoptic maps:

```yaml
data:
  geometry: spherical
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

If `data.samplers` is omitted, NF2 adds a `random_radial_grouped` sampler. If `data.validation` is omitted, NF2 adds a spherical validation grid.
