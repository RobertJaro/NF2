# FAQ

## How can I validate the quality of the extrapolations?

Use validation callbacks during training and run `nf2-metrics` after training. The most useful checks are divergence metrics, current-field alignment metrics such as `theta_J` and `sigma_J`, force-free residuals, magnetic-energy values, and visual inspection of boundary/slice plots.

```bash
nf2-metrics ./runs/case/extrapolation_result.nf2 --Mm_per_pixel 1.44 --height_range 0 80
```

Also compare exported fields and derived quantities such as `j`, `alpha`, and free energy over the region where the boundary data are reliable.

## How do I handle Out Of Memory errors?

Reduce the sampler and validation load before changing the model. Lower `data.sampler.batch_size`, boundary `batch_size`, validation `batch_size`, and validation resolution. For Cartesian runs, increasing `data.potential_boundary.strides` reduces potential-boundary memory. For post-training evaluation, pass a smaller `--batch_size` to `nf2-metrics`.

See [Training](training.md#out-of-memory-errors) for a step-by-step order.

## When is the extrapolation run converged?

Treat convergence as a combination of stable losses, stable quality metrics, and physically plausible validation plots. Boundary losses should stop improving rapidly, force-free and divergence metrics should flatten, and validation slices should no longer show large structural changes between validation intervals. For series runs, check that each step behaves consistently rather than judging only the final step.

If metrics are still improving at the end of training, increase `training.epochs` or `data.iterations`. If only one loss is dominating, tune its `weight` or schedule.

## How do I avoid using W&B online?

Set W&B to offline or disabled mode in the environment before launching NF2. Offline mode keeps local logs without syncing:

```bash
WANDB_MODE=offline nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml --run_path ./runs/case
```

Disabled mode turns W&B logging off more aggressively:

```bash
WANDB_MODE=disabled nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml --run_path ./runs/case
```

You can still keep a minimal `logging` block in the YAML for run names and local organization.
