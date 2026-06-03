# Extrapolations

NF2 extrapolations are usually run from YAML configuration files. The YAML file describes the run directory, input data, geometry, model, training settings, losses, callbacks, and export/evaluation datasets. The command-line tools load that YAML file, fill any placeholders, normalize the public schema, and then start the same Python training API used by scripts.

Choose the page that matches the geometry or run type:

- [Cartesian extrapolations](cartesian.md) cover local active-region boxes and multi-height Cartesian inputs.
- [Spherical extrapolations](spherical.md) cover full-disk and synoptic-map runs.
- [Analytical NLFF cases](analytical.md) provide fast benchmark and smoke-test runs.
- [Series runs](series.md) run time sequences after an initial single extrapolation has produced the starting NF2 state.

## YAML Configuration Files

A YAML configuration has a small set of top-level sections:

```yaml
path: ./runs/case
work_path: ./runs/case/work
logging:
  project: nf2
  name: "My extrapolation"
data:
  geometry: cartesian
  boundaries: []
training:
  epochs: 30
losses: []
callbacks: []
```

The required section is `data`. It must set `data.geometry` to `cartesian` or `spherical` and provide one or more boundary datasets. Everything else can be explicit or left to defaults. For a complete list of accepted keys, see the [Full YAML Reference](generated/configuration_reference.md) and [Dataset and Sampler Reference](generated/datasets_reference.md).

The command-line tools accept only `--config` as a named argument, then treat any additional `--name value` pairs as placeholder replacements. This is what lets one example YAML file run on different data files without editing the file each time.

```yaml
path: "<<run_path>>"
data:
  geometry: cartesian
  boundaries:
    - type: fits
      load_map: false
      files:
        Br: "<<Br>>"
        Bt: "<<Bt>>"
        Bp: "<<Bp>>"
```

Run it with custom files:

```bash
nf2-extrapolate \
  --config nf2/cartesian/minimal_fits.yaml \
  --run_path ./runs/ar377 \
  --Br ./data/ar377/Br.fits \
  --Bt ./data/ar377/Bt.fits \
  --Bp ./data/ar377/Bp.fits
```

You can also write literal paths directly in the YAML instead of placeholders:

```yaml
path: ./runs/ar377
data:
  geometry: cartesian
  boundaries:
    - type: fits
      load_map: false
      files:
        Br: ./data/ar377/Br.fits
        Bt: ./data/ar377/Bt.fits
        Bp: ./data/ar377/Bp.fits
```

Use placeholders for reusable example configs and literal paths for one-off local configs.

## What NF2 Does With The YAML

When a run starts, NF2 performs these steps:

1. Reads the YAML file.
2. Replaces `<<placeholder>>` values with matching command-line arguments.
3. Normalizes the public YAML schema into the internal runtime configuration.
4. Builds boundary, sampler, and validation datasets from `data`.
5. Builds the SIREN field model and configured losses.
6. Trains the model and writes `path/extrapolation_result.nf2`.

The `.nf2` result stores the trained model state and the normalized metadata needed by output helpers, exporters, and metrics.

## Python API

The command-line tools are thin wrappers around the Python API. Use `nf2.run(...)` when you want to create configurations programmatically:

```python
import nf2

nf2.run(
    path="./runs/case1",
    data={
        "geometry": "cartesian",
        "boundaries": [
            {
                "type": "fits",
                "load_map": False,
                "files": {
                    "Br": "./data/Br.fits",
                    "Bt": "./data/Bt.fits",
                    "Bp": "./data/Bp.fits",
                },
            }
        ],
        "z_range": [0, 80],
    },
    training={"epochs": 30},
)
```

After training, load the result with `nf2.load(...)`. NF2 chooses `CartesianOutput` or `SphericalOutput` from the checkpoint metadata:

```python
import nf2

out = nf2.load("./runs/case1/extrapolation_result.nf2")
cube = out.load_cube(height_range=[0, 80], Mm_per_pixel=1.0, metrics=["j", "alpha"])
```

Use `nf2.run_series(...)` for time series, `nf2.export_file(...)` for programmatic exports, and `from nf2.metrics import compute_metrics` for custom evaluation procedures.

```{toctree}
:maxdepth: 2
:caption: Extrapolations

cartesian
spherical
analytical
series
```
