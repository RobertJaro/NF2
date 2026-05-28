"""Public YAML configuration normalization for NF2."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_PATH = "./runs/nf2"


def normalize_config(config: dict) -> dict:
    """Normalize the public v0.4 YAML schema into the runtime config."""
    public_config = deepcopy(config)
    config = deepcopy(config)
    _reject_legacy_keys(config)

    path = config.pop("path", DEFAULT_PATH)
    work_path = config.pop("work_path", None)

    data = _normalize_data(config.pop("data", {}))
    model = _normalize_model(config.pop("model", None), data["type"])
    losses = _normalize_losses(config.pop("losses", None), data["type"], data)
    callbacks = _normalize_callbacks(config.pop("callbacks", None), data["type"], data)
    _validate_callback_refs(callbacks, data)

    normalized = {
        "path": path,
        "work_path": work_path,
        "logging": config.pop("logging", {}),
        "data": data,
        "model": model,
        "training": _normalize_training(config.pop("training", {})),
        "losses": losses,
        "transforms": _normalize_dataset_refs(config.pop("transforms", [])),
        "loss_scaling": config.pop("loss_scaling", _default_loss_scaling(data["type"], losses)),
        "callbacks": callbacks,
        "config": config,
    }
    if "meta_path" in normalized["config"]:
        normalized["meta_path"] = normalized["config"].pop("meta_path")
    if normalized["work_path"] is None:
        normalized.pop("work_path")
    normalized["config"] = {
        "schema_version": "0.4",
        "public": public_config,
        **normalized["config"],
    }
    return normalized


def _reject_legacy_keys(config):
    if "base_path" in config:
        raise ValueError("Config key 'base_path' was removed in v0.4. Use 'path'.")
    if "work_directory" in config:
        raise ValueError("Config key 'work_directory' was removed in v0.4. Use 'work_path'.")
    if "loss" in config:
        raise ValueError("Config key 'loss' was removed in v0.4. Use 'losses' with 'weight'.")
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    normalization = data.get("normalization", {}) or {}
    if "type" in data:
        raise ValueError("Config key 'data.type' was removed in v0.4. Use 'data.geometry'.")
    if "G_per_dB" in data or "G_per_dB" in normalization:
        raise ValueError("Config key 'G_per_dB' was removed in v0.4. Use 'Gauss_per_dB'.")
    if "length_unit_Mm" in data or "length_unit_Mm" in normalization:
        raise ValueError("Config key 'length_unit_Mm' was removed in v0.4. Use 'Mm_per_ds'.")
    if "field_unit_G" in data or "field_unit_G" in normalization:
        raise ValueError("Config key 'field_unit_G' was removed in v0.4. Use 'Gauss_per_dB'.")
    if "type" in model:
        raise ValueError("Config key 'model.type' was removed in v0.4. Use 'model.field'.")
    if "dim" in model:
        raise ValueError("Config key 'model.dim' was removed in v0.4. Use 'model.network.hidden_dim'.")
    if "dim" in (model.get("network", {}) or {}):
        raise ValueError("Config key 'model.network.dim' was removed in v0.4. Use 'model.network.hidden_dim'.")
    for loss in config.get("losses", []) or []:
        if "lambda" in loss:
            raise ValueError("Loss key 'lambda' was removed in v0.4. Use 'weight'.")


def _normalize_data(data):
    data = deepcopy(data)
    geometry = data.pop("geometry", None)
    if geometry not in {"cartesian", "spherical"}:
        raise ValueError("data.geometry must be explicitly set to 'cartesian' or 'spherical'.")

    if geometry == "cartesian":
        return _normalize_cartesian_data(data)
    return _normalize_spherical_data(data)


def _normalize_cartesian_data(data):
    normalization = data.pop("normalization", {})
    mm_per_ds = normalization.pop("Mm_per_ds", data.pop("Mm_per_ds", 100))
    g_per_db = normalization.pop("Gauss_per_dB", data.pop("Gauss_per_dB", 1000))

    boundaries = data.pop("boundaries", None)
    if boundaries is None:
        raise ValueError("Cartesian configs require data.boundaries with at least one boundary dataset.")
    boundaries = [_normalize_dataset_config(boundary, role="boundary") for boundary in _as_list(boundaries)]

    sampler = data.pop("sampler", None)
    samplers = data.pop("samplers", None)
    if sampler is None and samplers:
        sampler = _as_list(samplers)[0]
    sampler = _normalize_sampler(sampler or {"type": "height", "id": "random"})

    potential_boundary = data.pop("potential_boundary", {"type": "potential", "id": "potential", "strides": 4})
    potential_boundary = _normalize_potential_boundary(potential_boundary)

    validation = data.pop("validation", None)
    validation_datasets = None
    if validation is not None:
        validation_datasets = [_normalize_dataset_config(v, role="validation") for v in _as_list(validation)]

    normalized = {
        "type": "cartesian",
        "boundaries": boundaries,
        "sampler": sampler,
        "potential_boundary": potential_boundary,
        "Mm_per_ds": mm_per_ds,
        "Gauss_per_dB": g_per_db,
        **data,
    }
    if validation_datasets is not None:
        normalized["validation"] = validation_datasets
    return normalized


def _normalize_spherical_data(data):
    normalization = data.pop("normalization", {})
    mm_per_ds = normalization.pop("Mm_per_ds", data.pop("Mm_per_ds", 100))
    g_per_db = normalization.pop("Gauss_per_dB", data.pop("Gauss_per_dB", 1000))
    iterations = data.pop("iterations", None)

    boundaries = [_normalize_dataset_config(b, role="boundary") for b in _as_list(data.pop("boundaries", []))]
    samplers = [_normalize_dataset_config(s, role="sampler") for s in _as_list(data.pop("samplers", []))]
    if not boundaries:
        raise ValueError("Spherical configs require data.boundaries with at least one boundary dataset.")
    if not samplers:
        samplers = [_normalize_dataset_config({"id": "random", "type": "random_radial_grouped"}, role="sampler")]
    if iterations is not None:
        for sampler in samplers:
            if sampler.get("type") in {"random_spherical", "random_radial_grouped"}:
                sampler.setdefault("length", iterations)

    validation = data.pop("validation", None)
    if validation is None:
        validation = [{"id": "sphere", "type": "sphere"}]
    validation_configs = [_normalize_dataset_config(v, role="validation") for v in _as_list(validation)]

    return {
        "type": "spherical",
        "boundaries": boundaries,
        "samplers": samplers,
        "validation": validation_configs,
        "Mm_per_ds": mm_per_ds,
        "Gauss_per_dB": g_per_db,
        **data,
    }


def _normalize_model(model, geometry):
    model = deepcopy(model or {})
    field = model.pop("field", "vector_potential")
    if field not in {"b", "vector_potential"}:
        raise ValueError("model.field must be 'b' or 'vector_potential'.")
    network = model.pop("network", {}) or {}
    network_type = network.pop("type", "siren")
    if network_type != "siren":
        raise ValueError("Only SIREN networks are supported in v0.4.")

    hidden_dim = network.pop("hidden_dim", model.pop("hidden_dim", 512 if geometry == "spherical" else 256))
    normalized = {"type": field, "dim": hidden_dim, **network, **model}
    if "w0_initial" in normalized:
        normalized["w0_init"] = normalized.pop("w0_initial")
    if "layers" in normalized:
        normalized["n_layers"] = normalized.pop("layers")
    return normalized


def _normalize_training(training):
    return {
        "epochs": 10,
        "optimizer": {"start": 5e-4, "end": 5e-5, "iterations": 1e5},
        "trainer": {},
        **deepcopy(training),
    }


def _normalize_losses(losses, geometry, data):
    if losses is None:
        losses = _default_losses(geometry, data)
    result = []
    for loss in deepcopy(losses):
        if "lambda" in loss:
            raise ValueError("Loss key 'lambda' was removed in v0.4. Use 'weight'.")
        if "weight" not in loss:
            raise ValueError(f"Loss '{loss.get('name', loss.get('type', '<unknown>'))}' requires a 'weight'.")
        if "datasets" in loss:
            loss["ds_id"] = loss.pop("datasets")
        result.append(loss)
    return result


def _default_losses(geometry, data):
    if geometry == "cartesian":
        boundary_ids = [cfg["id"] for cfg in data["boundaries"]]
        losses = [
            {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": boundary_ids},
            {"type": "force_free", "name": "force_free", "weight": 1.0e-4, "datasets": ["random"]},
        ]
        if data.get("potential_boundary", {}).get("type") not in {None, "none"}:
            losses.insert(1, {"type": "boundary", "name": "potential_boundary", "weight": 10.0, "datasets": "potential"})
            losses.append({"type": "potential", "name": "potential", "weight": {"type": "step", "steps": 5000, "start": 1.0e-4, "end": 0.0}, "datasets": ["random"]})
        return losses
    boundary_ids = [cfg["id"] for cfg in data["boundaries"] if cfg["type"] == "map"]
    losses = [
        {"type": "force_free", "name": "force_free", "weight": {"start": 1.0e-4, "end": 1.0e-2, "iterations": 50000}, "datasets": ["random"]},
        {"type": "potential", "name": "potential", "weight": {"start": 1.0e-4, "end": 1.0e-2, "iterations": 50000}, "datasets": ["random"]},
    ]
    if boundary_ids:
        losses.insert(0, {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": boundary_ids})
    return losses


def _default_loss_scaling(geometry, losses):
    loss_ids = [loss.get("name", loss["type"]) for loss in losses if loss.get("type") in {"force_free", "potential", "energy_gradient"}]
    if not loss_ids:
        return []
    if geometry == "cartesian":
        return [{"type": "b_height", "name": "b_height", "loss_ids": loss_ids}]
    return [{"type": "radial", "name": "radial", "base_radius": 1.0, "loss_ids": loss_ids}]


def _normalize_callbacks(callbacks, geometry, data):
    if callbacks is None:
        callbacks = _default_callbacks(geometry, data)
    normalized = []
    for callback in deepcopy(callbacks):
        if "dataset" in callback:
            callback["ds_id"] = callback.pop("dataset")
        normalized.append(callback)
    return normalized


def _validate_callback_refs(callbacks, data):
    validation_ids = _validation_ids(data)
    for callback in callbacks:
        ds_id = callback.get("ds_id")
        if ds_id is None or ds_id in validation_ids:
            continue
        raise ValueError(
            f"Callback dataset '{ds_id}' is not defined in data.validation. "
            f"Available validation datasets: {sorted(validation_ids)}"
        )


def _validation_ids(data):
    if data["type"] == "cartesian" and "validation" not in data:
        return {cfg["id"] for cfg in data["boundaries"]} | {"cube", "slices"}
    return {cfg["id"] for cfg in data.get("validation", [])}


def _default_callbacks(geometry, data):
    if geometry == "cartesian":
        boundary_id = data["boundaries"][0]["id"]
        return [
            {"type": "boundary", "dataset": boundary_id},
            {"type": "metrics", "dataset": "cube"},
            {"type": "slices", "dataset": "slices"},
        ]
    validation_ids = [cfg["id"] for cfg in data.get("validation", [])]
    callbacks = []
    if "full_disk_valid" in validation_ids:
        callbacks.append({"type": "boundary", "dataset": "full_disk_valid"})
    if "sphere" in validation_ids:
        callbacks.append({"type": "metrics", "dataset": "sphere"})
    if "slices" in validation_ids:
        callbacks.append({"type": "spherical_slices", "dataset": "slices"})
    return callbacks


def _normalize_dataset_config(config, role):
    config = deepcopy(config)
    if role == "sampler" and "iterations" in config:
        if "length" in config:
            raise ValueError("Sampler keys 'length' and 'iterations' are both set. Use data.iterations.")
        config["length"] = config.pop("iterations")
    if "dataset" in config:
        config["id"] = config.pop("dataset")
    if "files" in config and config.get("type") in {"fits", "sharp", "los_trv_azi", "los", "fld_inc_azi"}:
        config["fits_path"] = config.pop("files")
    if "errors" in config:
        errors = config.pop("errors")
        if config.get("type") == "map":
            config.setdefault("files", {}).update(errors)
        else:
            config["error_path"] = errors
    if role in {"boundary", "validation", "sampler"} and "id" not in config:
        if role == "boundary":
            config["id"] = "boundary"
        elif role == "sampler":
            config["id"] = "random"
        else:
            config["id"] = f"valid_{config['type']}"
    return config


def _normalize_sampler(config):
    config = deepcopy(config)
    config.pop("id", None)
    if config.get("type") == "random_height":
        config["type"] = "height"
    return config


def _normalize_potential_boundary(config):
    if config is None:
        return {"type": "none"}
    config = deepcopy(config)
    config.pop("id", None)
    return config


def _normalize_dataset_refs(configs):
    normalized = []
    for config in deepcopy(configs):
        if "datasets" in config:
            config["ds_id"] = config.pop("datasets")
        normalized.append(config)
    return normalized


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
