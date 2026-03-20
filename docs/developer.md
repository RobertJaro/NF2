# Developer Guide

This page describes where to extend the refactored framework.

## Core Extension Points

### Add a New Geometry-Aware Execution Hook

Look at:

- `nf2/core/adapters.py`
- `nf2/core/runner.py`

### Add a New Dataset Type

Add it in the geometry-specific loader implementation, not in the core runner.

Typical places:

- `nf2/loader/cartesian.py`
- `nf2/loader/spherical.py`

### Add a New Export Format

Add a writer in:

- `nf2/export/writers.py`

Then wire it into:

- `nf2/export/core.py`
- `nf2/export/cli.py`

### Add a New Metric

Add the metric function in:

- `nf2/evaluation/output_metrics.py`

If it should be part of the reusable interface, expose it through:

- `nf2/eval/metrics.py`

## Design Principle

Keep:

- geometry-specific implementation separate
- execution/config/export seams shared

Do not reintroduce duplicated cartesian vs spherical runner logic.
