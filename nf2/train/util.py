import yaml
import warnings

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
    assert all([k.startswith('--') for k in overwrite_args[::2]]), \
        'Only accept --config and overwrite arguments (must start with --)'
    overwrite_args = {k.replace('--', ''): v for k, v in zip(overwrite_args[::2], overwrite_args[1::2])}
    with open(yaml_config_file) as f:
        config_str = f.read()
    for overwrite_key, overwrite_value in overwrite_args.items():
        config_str = config_str.replace('<<%s>>' % overwrite_key, overwrite_value)
    config = yaml.safe_load(config_str)
    return normalize_config(config)
