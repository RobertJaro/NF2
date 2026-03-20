# Losses and Transforms

NF2 separates model definition from physics/data constraints.

## Common Losses

Frequently used losses in repository configs include:

- `boundary`
- `boundary_los_trv_azi`
- `force_free`
- `divergence`
- `potential`
- `energy_gradient`
- `height`
- `radial`

Typical loss fields:

- `type`
- `lambda`
- `ds_id`
- `name`

## Transforms

Transform modules are used when coordinate mappings or ambiguity handling are part of the training setup.

Common transform types:

- `height`
- `height_range`
- `optical_depth`
- `azimuth`

## Loss Scaling

Scaling modules adjust selected losses dynamically.

Common types:

- `exponential`
- `potential_fit`
- `b_height`
- `radial`

## Practical Guidance

- Keep `boundary` strong when matching observations matters most.
- Increase `force_free` when moving toward more force-free solutions.
- Use geometry-specific random/interior datasets for force-free and potential terms.
- Be explicit about dataset IDs when multiple training boundaries are present.
