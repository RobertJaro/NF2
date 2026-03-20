# Architecture

The refactor is organized around a small shared core plus separate geometry implementations.

## Core

- `nf2/core/runner.py`: shared execution path
- `nf2/core/adapters.py`: adapter interface and registry
- `nf2/config/schema.py`: canonical config schema loader
- `nf2/output/`: checkpoint and geometry-aware output adapters
- `nf2/export/`: shared exporters and CLI

## Geometry

- `nf2/geometry/cartesian/`
- `nf2/geometry/spherical/`

These packages keep their own data-module and geometry-specific behavior, while exposing a narrow interface to the core.

## Evaluation

Reusable evaluation entrypoints are exposed through:

- `nf2.eval.metrics`
- `nf2.eval.outputs`
- `nf2.eval.checkpoints`
