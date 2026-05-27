# Installing NF2

NF2 supports installation through pip and conda.

## Pip

Install from a local checkout:

```bash
python -m pip install .
```

Install for development:

```bash
python -m pip install -e ".[wandb,jsoc,pfss,docs,dev]"
```

Build release artifacts:

```bash
python -m pip install build
python -m build
```

The build creates:

- `dist/nf2-0.4.0.tar.gz`
- `dist/nf2-0.4.0-py3-none-any.whl`

## Conda Environment

Create the recommended development environment:

```bash
conda env create -f environment.yml
conda activate nf2
```

The environment installs compiled scientific dependencies through conda and installs NF2 itself in editable mode with `pip --no-deps`.

## Conda Package Recipe

Render the recipe:

```bash
CONDA_BLD_PATH=/tmp/conda-bld conda render conda-recipe
```

Build the recipe:

```bash
CONDA_BLD_PATH=/tmp/conda-bld conda build conda-recipe
```

The recipe lives in `conda-recipe/meta.yaml` and exposes the public command-line tools:

- `nf2-extrapolate`
- `nf2-extrapolate-series`
- `nf2-export`
- `nf2-metrics`
- `nf2-download`
- `nf2-noaa-to-sharp`

## Smoke Test

```bash
python - <<'PY'
import nf2
import torch
import lightning

print("NF2:", nf2.__version__)
print("Torch:", torch.__version__)
print("Lightning:", lightning.__version__)
PY

nf2-export --help
nf2-extrapolate --help
```
