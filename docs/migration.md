# Migration

The repository has moved to a clean canonical configuration format.

Key changes:

- `base_path` -> `run.output_dir`
- `work_directory` -> `run.work_dir`
- `meta_path` -> `run.resume_from`
- `data.slices` / `data.train_configs` -> `data.train`
- `data.validation_configs` / `data.valid_configs` -> `data.validation`
- `training.loss_config` / top-level `loss` -> top-level `losses`

The repository YAML files have already been migrated. New configs should use only the canonical format.
