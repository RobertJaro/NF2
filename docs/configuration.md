# Configuration

NF2 now uses one canonical configuration schema for both geometries.

```yaml
run:
  mode: single
  geometry: cartesian
  output_dir: /path/to/results
  work_dir: /path/to/work

logging:
  project: nf2
  name: demo

data:
  parameters:
    iterations: 10000
    num_workers: 8
  train:
    - type: fits
      fits_path:
        Br: /path/to/Br.fits
        Bt: /path/to/Bt.fits
        Bp: /path/to/Bp.fits
  validation:
    - type: cube
      ds_id: cube

model:
  type: vector_potential
  dim: 256

training:
  epochs: 20

losses:
  - type: boundary
    lambda: 1.0
    ds_id: boundary_01
```

Notes:

- `run.geometry` chooses the adapter: `cartesian` or `spherical`
- `data.parameters` stores shared loader parameters
- `data.train` and `data.validation` hold dataset definitions
- series runs use `run.mode: series` and `data.sequence`
