# Installing NF2

NF2 targets Python 3.11 and 3.12.

## Install from PyPI

### pip

Create and activate a virtual environment, then install NF2:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install nf2
```

On Windows, use `.venv\Scripts\activate.bat` in Command Prompt or `.venv\Scripts\Activate.ps1` in PowerShell.

### uv

Create and activate a Python 3.12 virtual environment, install the PyTorch build that matches the available hardware, then install NF2:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --torch-backend=auto
uv pip install nf2
```

## PyTorch and CUDA

NF2 declares compatible `torch` and `torchvision` dependencies, so a normal installation includes their default builds. To use a specific CUDA build, select the appropriate command at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and install PyTorch before installing NF2.

For example, install the CUDA 12.6 wheels with:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Or install the most recent default PyTorch build with:

```bash
python -m pip install torch torchvision
```

These are examples only. Use the PyTorch selector above to obtain the command for your operating system and required CUDA version.
This is not necessary if using the uv installation described above as a compatible build will be automatically installed.

## Development installation

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-user>/NF2.git
cd NF2
git remote add upstream https://github.com/RobertJaro/NF2.git
```

Choose one of the following development environments. Each installs NF2 in editable mode with the documentation, test, lint, and packaging tools.

### pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### uv

```bash
uv sync --all-extras
source .venv/bin/activate
```

### Conda

```bash
conda env create -f environment.yml
conda activate nf2
```

Build the documentation locally before opening a pull request:

```bash
LC_ALL=C LANG=C SUNPY_CONFIGDIR=/tmp/sunpy MPLCONFIGDIR=/tmp/matplotlib sphinx-build -b html docs docs/_build/html

```

## Packaging

Build a source distribution and wheel:

```bash
python -m build
```

With uv, use:

```bash
uv build
```

To render or build the Conda package recipe, first install `conda-build`:

```bash
conda install -c conda-forge conda-build
conda render conda-recipe
conda build conda-recipe
```
