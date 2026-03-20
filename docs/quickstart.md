# Quickstart

This page is the fastest path from clone to first result.

## 1. Install

```bash
pip install -e .
```

Make sure PyTorch is installed in the runtime you plan to use for training or export.

## 2. Pick a Config

Representative config families live in [`config/`](/Users/rjarolim/PycharmProjects/NF2/config):

- `config/sharp/`: standard cartesian SHARP workflows
- `config/spherical/`: spherical full-disk and synoptic workflows
- `config/topology/`: multi-slice and topology-driven experiments
- `config/magnetostatic/`: magnetostatic models

## 3. Run Training

Single run:

```bash
nf2-extrapolate --config /absolute/path/to/config.yaml
```

Series run:

```bash
nf2-extrapolate-series --config /absolute/path/to/series.yaml
```

## 4. Export a Result

```bash
nf2-export --checkpoint /absolute/path/to/result.nf2 --format vtk
```

Or export a whole series:

```bash
nf2-export --series --checkpoint "/absolute/path/to/results/*.nf2" --format hdf5 --out_dir /absolute/path/to/exports
```

## 5. Understand the Config

The new canonical config shape is:

```yaml
run:
  mode: single
  geometry: cartesian
  output_dir: /path/to/results
  work_dir: /path/to/work

data:
  parameters: {}
  train: []
  validation: []

model: {}
training: {}
losses: []
```

Read [Configuration](configuration.md) next.
