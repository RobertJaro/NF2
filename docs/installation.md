# Installation

Create the recommended conda environment for development and examples:

```bash
conda env create -f environment.yml
conda activate nf2
```

For development, install directly into an existing environment:

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

Build a source distribution and wheel:

```bash
python -m pip install build
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

NF2 targets Python 3.11/3.12 with current PyTorch, Lightning, and W&B releases.

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
