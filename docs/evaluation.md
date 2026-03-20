# Evaluation

NF2 currently contains two evaluation layers:

1. reusable API surface for framework code
2. many experiment- and paper-specific scripts under `nf2/evaluation/`

## Reusable API

Use the lightweight facade in `nf2.eval`:

- `nf2.eval.checkpoints.open_checkpoint`
- `nf2.eval.outputs.load_output_adapter`
- `nf2.eval.outputs.sample_checkpoint`
- `nf2.eval.metrics.METRICS`

These wrap the shared checkpoint and output layers added during the refactor.

## Metric Surface

Metrics are currently sourced from `nf2.evaluation.output_metrics`.

Examples include:

- `j`
- `alpha`
- `energy`
- `free_energy`
- `spherical_energy_gradient`

## Legacy Script Tree

The existing `nf2/evaluation/` tree still contains many project-specific analysis scripts for:

- `sharp`
- `spherical`
- `sst`
- `muram`
- `analytical`
- `solis`

Those scripts remain useful, but they should be treated as case-study tooling rather than stable framework APIs.
