from nf2.output.checkpoint import NF2Checkpoint
from nf2.output.adapters import CartesianOutputAdapter, SphericalOutputAdapter


def load_output_adapter(path, device=None):
    checkpoint = NF2Checkpoint(path, device=device)
    if checkpoint.geometry == "cartesian":
        return CartesianOutputAdapter(path, device=device)
    if checkpoint.geometry == "spherical":
        return SphericalOutputAdapter(path, device=device)
    raise ValueError(f"Unsupported geometry: {checkpoint.geometry}")


def sample_checkpoint(path, device=None, **kwargs):
    return load_output_adapter(path, device=device).sample(**kwargs)
