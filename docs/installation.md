# Installation

## Pip Installation

Install NF2 from PyPI:

```bash
pip install nf2
```

Install optional dependency groups when you need W&B logging, JSOC downloads, PFSS support, documentation tools, or development tools:

```bash
pip install "nf2[wandb]"
pip install "nf2[jsoc,pfss]"
pip install "nf2[docs]"
```

## Conda Installation

Install NF2 with conda:

```bash
conda install nf2
```

Create a fresh environment if you prefer to isolate NF2:

```bash
conda create -n nf2 python=3.11 nf2
conda activate nf2
```

If your conda setup does not already use conda-forge, add it or pass `-c conda-forge`.

## Local Installation

Create the recommended conda environment from a local source checkout:

```bash
conda env create -f environment.yml
conda activate nf2
```

Install NF2 from the checkout:

```bash
python -m pip install -e ".[wandb,jsoc,pfss,docs,dev]"
```

For a regular pip install from a local checkout:

```bash
python -m pip install .
```

Optional dependency groups:

```bash
python -m pip install ".[wandb]"
python -m pip install ".[docs]"
python -m pip install ".[wandb,jsoc,pfss]"
```

NF2 targets Python 3.11/3.12 with current PyTorch, Lightning, and W&B releases.

Verify the installation:

```bash
python - <<'PY'
import torch
import lightning
import nf2

print("NF2:", nf2.__version__)
print("Torch:", torch.__version__)
print("Lightning:", lightning.__version__)
print("CUDA:", torch.cuda.is_available())
PY
```

## Development Installation

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-user>/NF2.git
cd NF2
git remote add upstream https://github.com/RobertJaro/NF2.git
```

Create the development environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate nf2
python -m pip install -e ".[wandb,jsoc,pfss,docs,dev]"
```

Keep your fork current with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

Create a feature branch for changes:

```bash
git checkout -b my-docs-or-feature-change
```

Build the documentation locally before opening a pull request:

```bash
LC_ALL=C LANG=C SUNPY_CONFIGDIR=/tmp/sunpy MPLCONFIGDIR=/tmp/matplotlib sphinx-build -b html docs docs/_build/html
```

## Packaging

Build a source distribution and wheel:

```bash
python -m pip install build
python -m build
```

When using conda for packaging tools, install the conda-forge package name:

```bash
conda install -c conda-forge python-build twine
python -m build
```

For conda packaging, use the project dependencies from `pyproject.toml` and the environment pins in `environment.yml` as the source of truth. A minimal user-facing conda environment can be created with:

```bash
conda env create -f environment.yml
```

Render or build the conda package recipe:

```bash
CONDA_BLD_PATH=/tmp/conda-bld conda render conda-recipe
CONDA_BLD_PATH=/tmp/conda-bld conda build conda-recipe
```
