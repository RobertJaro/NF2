from __future__ import annotations

from copy import deepcopy

import yaml


def _read_yaml_with_overrides(yaml_config_file, overwrite_args=None):
    overwrite_args = [] if overwrite_args is None else overwrite_args
    assert all([k.startswith("--") for k in overwrite_args[::2]]), (
        "Only accept --config and overwrite arguments (must start with --)"
    )
    overwrite_args = {k.replace("--", ""): v for k, v in zip(overwrite_args[::2], overwrite_args[1::2])}
    with open(yaml_config_file) as f:
        config_str = f.read()
    for overwrite_key, overwrite_value in overwrite_args.items():
        config_str = config_str.replace("<<%s>>" % overwrite_key, overwrite_value)
    return yaml.safe_load(config_str)


def canonical_to_runtime_config(config):
    config = deepcopy(config)
    if "run" not in config:
        raise ValueError("Config must use the canonical schema with a top-level 'run' section")

    run = config["run"]
    data_cfg = deepcopy(config.get("data", {}))
    parameters = deepcopy(data_cfg.get("parameters", {}))
    mode = run["mode"]
    geometry = run["geometry"]

    runtime = {
        "base_path": run["output_dir"],
        "work_directory": run.get("work_dir"),
        "logging": deepcopy(config.get("logging", {})),
        "data": parameters,
        "model": deepcopy(config.get("model", {})),
        "training": deepcopy(config.get("training", {})),
        "loss": deepcopy(config.get("losses", [])),
        "callbacks": deepcopy(config.get("callbacks", [])),
        "transforms": deepcopy(config.get("transforms", [])),
        "loss_scaling": deepcopy(config.get("loss_scaling", [])),
        "config": config,
    }

    train = deepcopy(data_cfg.get("train", []))
    validation = deepcopy(data_cfg.get("validation", []))
    sequence = deepcopy(data_cfg.get("sequence"))

    if geometry == "cartesian":
        runtime["data"]["type"] = "cartesian"
        if mode == "single":
            runtime["data"]["train_configs"] = train
            if validation:
                runtime["data"]["valid_configs"] = validation
        elif mode == "series":
            if not isinstance(sequence, dict) or "frames" not in sequence:
                raise ValueError("Cartesian series config requires data.sequence.frames")
            frames = deepcopy(sequence["frames"])
            runtime["data"]["train_configs"] = [frames] if isinstance(frames, dict) else frames
            if validation:
                runtime["data"]["valid_configs"] = validation
            runtime["meta_path"] = run["resume_from"]
        else:
            raise ValueError(f"Unsupported run mode: {mode}")
    elif geometry == "spherical":
        runtime["data"]["type"] = "spherical"
        if mode == "single":
            runtime["data"]["train_configs"] = train
            runtime["data"]["validation_configs"] = validation
        elif mode == "series":
            if not isinstance(sequence, dict) or "frames" not in sequence or "synoptic" not in sequence:
                raise ValueError("Spherical series config requires data.sequence.frames and data.sequence.synoptic")
            runtime["data"]["fits_paths"] = sequence["frames"]
            runtime["data"]["synoptic_fits_path"] = sequence["synoptic"]
            runtime["data"]["validation_configs"] = validation
            runtime["meta_path"] = run["resume_from"]
        else:
            raise ValueError(f"Unsupported run mode: {mode}")
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    return runtime


def load_yaml_config(yaml_config_file, overwrite_args=None):
    config = _read_yaml_with_overrides(yaml_config_file, overwrite_args=overwrite_args)
    return canonical_to_runtime_config(config)
