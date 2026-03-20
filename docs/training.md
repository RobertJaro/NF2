# Training

Single-run training:

```bash
nf2-extrapolate --config /path/to/config.yaml
```

Series training:

```bash
nf2-extrapolate-series --config /path/to/series.yaml
```

The shared runner lives in `nf2/core/runner.py` and dispatches to geometry-specific data modules through the adapter registry.

Internally:

- `run.mode` controls single vs series execution
- `run.geometry` selects the geometry adapter
- the data module still stays geometry-specific
