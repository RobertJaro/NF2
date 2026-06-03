# Installing NF2

NF2 supports installation through pip and conda.

## Pip

Install from PyPI:

```bash
pip install nf2
```

## PyTorch And CUDA

For GPU-enabled PyTorch installs, use the official PyTorch selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Select your CUDA version and run the generated command before installing or running NF2.

For CUDA 12.6:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For the most recent default PyTorch build:

```bash
pip3 install torch torchvision
```

Install from a local checkout:

```bash
python -m pip install .
```

Install for development:

```bash
python -m pip install -r requirements.txt
```

Build release artifacts:

```bash
python -m pip install build
python -m build
```

When using conda for packaging tools, install the conda-forge package name:

```bash
conda install -c conda-forge python-build twine
python -m build
```

The build creates:

- `dist/nf2-0.4.1.tar.gz`
- `dist/nf2-0.4.1-py3-none-any.whl`

## Conda Environment

Install with conda:

```bash
conda install nf2
```

Create a fresh NF2 environment:

```bash
conda create -n nf2 python=3.11 nf2
conda activate nf2
```

If your conda setup does not already use conda-forge, add it or pass `-c conda-forge`.

Create the recommended development environment from a local checkout:

```bash
conda env create -f environment.yml
conda activate nf2
```

The environment uses conda for Python isolation and installs the NF2 runtime, documentation, and development stack with pip. The pip section is explicit so key packages such as PyTorch, Lightning, and SunPy are visible, while avoiding conda channel availability issues for pinned Python packages.

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
print("CUDA:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"CUDA device {idx}:", torch.cuda.get_device_name(idx))
PY

nf2-export --help
nf2-extrapolate --help
```
