# Installation

Install the package with:

```bash
pip install nf2
```

For development installs from the repository root:

```bash
pip install -e .
```

NF2 expects PyTorch to be available in the target environment.

Useful commands:

```bash
nf2-extrapolate --config /path/to/config.yaml
nf2-extrapolate-series --config /path/to/config.yaml
nf2-export --checkpoint /path/to/result.nf2 --format vtk
```

## Build Documentation Locally

Install the docs dependencies:

```bash
pip install -r docs/requirements.txt
```

Build the site:

```bash
make docs
```

Serve a local preview:

```bash
make docs-serve
```
