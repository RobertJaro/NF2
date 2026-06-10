import re
import yaml
import warnings

from importlib import resources
from pathlib import Path

from nf2.train.config import normalize_config


def suppress_runtime_warnings():
    import torch

    setter = getattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None)
    if setter is not None:
        setter(False)
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=UserWarning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )


def suppress_accumulate_grad_stream_warning():
    suppress_runtime_warnings()


def is_interactive_environment():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def load_yaml_config(yaml_config_file, overwrite_args=None):
    overwrite_args = [] if overwrite_args is None else overwrite_args
    overwrite_args = _parse_overwrite_args(overwrite_args)
    config_str = _read_yaml_config(yaml_config_file)
    config_str = _resolve_placeholders(config_str, overwrite_args)
    config = yaml.safe_load(config_str)
    _drop_unresolved_optional_errors(config)
    _reject_unresolved_placeholders(config)
    return normalize_config(config)


_PLACEHOLDER_PATTERN = re.compile(r"<<([^<>;]+)(?:;([^<>]*))?>>")


def _resolve_placeholders(config_str, overwrite_args):
    def replace(match):
        key = match.group(1).strip()
        default = match.group(2)
        if key in overwrite_args:
            return _format_placeholder_value(key, overwrite_args[key])
        if default is not None:
            return _format_placeholder_value(key, default.strip())
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(replace, config_str)


def _format_placeholder_value(key, value):
    if isinstance(value, list):
        return f"[{', '.join(str(v) for v in value)}]"
    return str(value)


def _parse_overwrite_args(overwrite_args):
    parsed = {}
    i = 0
    while i < len(overwrite_args):
        key = overwrite_args[i]
        if not key.startswith("--"):
            raise ValueError("Only accept --config and overwrite arguments (must start with --)")
        key = key.replace("--", "", 1)
        values = []
        i += 1
        while i < len(overwrite_args) and not overwrite_args[i].startswith("--"):
            values.append(overwrite_args[i])
            i += 1
        if not values:
            raise ValueError(f"--{key} requires a value")
        parsed[key] = values[0] if len(values) == 1 else values
    return parsed


def _reject_unresolved_placeholders(config):
    placeholder = _find_unresolved_placeholder(config)
    if placeholder is not None:
        raise ValueError(f"Missing required config argument for placeholder '{placeholder}'.")


def _find_unresolved_placeholder(value):
    if isinstance(value, str):
        match = _PLACEHOLDER_PATTERN.search(value)
        if match is not None:
            return match.group(1).strip()
    if isinstance(value, dict):
        for child in value.values():
            placeholder = _find_unresolved_placeholder(child)
            if placeholder is not None:
                return placeholder
    if isinstance(value, list):
        for child in value:
            placeholder = _find_unresolved_placeholder(child)
            if placeholder is not None:
                return placeholder
    return None


def _read_yaml_config(yaml_config_file):
    yaml_config_path = Path(yaml_config_file)
    if yaml_config_path.exists():
        return yaml_config_path.read_text()

    template_path = _resolve_bundled_config(yaml_config_file)
    if template_path is not None:
        return template_path.read_text()

    raise FileNotFoundError(
        f"Config file '{yaml_config_file}' was not found. "
        "Use a local YAML path or a bundled template such as 'nf2/cartesian/sharp_cea.yaml'."
    )


def _resolve_bundled_config(yaml_config_file):
    config_name = str(yaml_config_file).strip().replace("\\", "/")
    for prefix in ("examples/configs/", "nf2/configs/", "nf2/"):
        if config_name.startswith(prefix):
            config_name = config_name[len(prefix):]

    if Path(config_name).is_absolute() or ".." in Path(config_name).parts:
        return None

    templates = resources.files("nf2.configs")
    candidates = [templates.joinpath(config_name)]
    if "/" not in config_name:
        candidates.extend(templates.joinpath(group, config_name) for group in ("benchmark", "cartesian", "spherical"))

    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def _drop_unresolved_optional_errors(config):
    dropped_error_ids = set()
    _drop_unresolved_optional_errors_inplace(config, dropped_error_ids)
    if dropped_error_ids:
        _drop_dependent_optional_error_references(config, dropped_error_ids)


def _drop_unresolved_optional_errors_inplace(config, dropped_error_ids):
    if isinstance(config, dict):
        errors = config.get("errors")
        if errors is not None and _contains_unresolved_placeholder(errors):
            if config.get("id") is not None:
                dropped_error_ids.add(config["id"])
            warnings.warn(
                "Skipping optional error-file configuration because one or more error placeholders were not set.",
                UserWarning,
                stacklevel=2,
            )
            config.pop("errors")
        for value in config.values():
            _drop_unresolved_optional_errors_inplace(value, dropped_error_ids)
    elif isinstance(config, list):
        for value in config:
            _drop_unresolved_optional_errors_inplace(value, dropped_error_ids)


def _drop_dependent_optional_error_references(config, dropped_error_ids):
    if isinstance(config, dict):
        errors = config.get("errors")
        if errors is not None and _references_dropped_errors(errors, dropped_error_ids):
            warnings.warn(
                "Skipping optional error-file references because their source error configuration was skipped.",
                UserWarning,
                stacklevel=2,
            )
            config.pop("errors")
        for value in config.values():
            _drop_dependent_optional_error_references(value, dropped_error_ids)
    elif isinstance(config, list):
        for value in config:
            _drop_dependent_optional_error_references(value, dropped_error_ids)


def _contains_unresolved_placeholder(value):
    if isinstance(value, str):
        return "<<" in value and ">>" in value
    if isinstance(value, dict):
        return any(_contains_unresolved_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved_placeholder(v) for v in value)
    return False


def _references_dropped_errors(value, dropped_error_ids):
    if isinstance(value, str):
        return any(f"[[{dataset_id}.errors." in value for dataset_id in dropped_error_ids)
    if isinstance(value, dict):
        return any(_references_dropped_errors(v, dropped_error_ids) for v in value.values())
    if isinstance(value, list):
        return any(_references_dropped_errors(v, dropped_error_ids) for v in value)
    return False
