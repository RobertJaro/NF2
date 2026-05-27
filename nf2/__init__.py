"""Public access points for the NF2 framework.

The top-level package intentionally avoids importing heavy astronomy and deep
learning dependencies until a caller asks for the matching helper.
"""

__version__ = "0.4.0"


def load(path, device=None):
    """Load an NF2 result and return the matching output helper.

    Parameters
    ----------
    path:
        Path to a ``.nf2`` checkpoint.
    device:
        Optional PyTorch device used for evaluation.

    Returns
    -------
    CartesianOutput or SphericalOutput
        Geometry-specific output helper selected from checkpoint metadata.
    """
    import torch
    from nf2.evaluation.output import CartesianOutput, SphericalOutput

    state = torch.load(path, map_location="cpu", weights_only=False)
    geometry = state.get("data", {}).get("type")
    if geometry == "cartesian":
        return CartesianOutput(path, device=device)
    if geometry == "spherical":
        return SphericalOutput(path, device=device)
    raise ValueError(f"Unsupported NF2 geometry: {geometry!r}")


def run(*args, **kwargs):
    """Run a single NF2 extrapolation.

    The function accepts either the normalized runtime configuration used by
    :mod:`nf2.extrapolate` or the public v0.4 YAML-style schema. When the
    public schema is passed, it is normalized before training starts.
    """
    data = kwargs.get("data", {})
    if data.get("geometry") is not None or ("data" in kwargs and "boundaries" not in data):
        from nf2.train.config import normalize_config

        kwargs = normalize_config(kwargs)
    from nf2.extrapolate import run as _run

    return _run(*args, **kwargs)


def run_series(*args, **kwargs):
    """Run an NF2 extrapolation series.

    Accepts the same public configuration schema as :func:`run`, with series
    placeholders expanded before the data module is built.
    """
    data = kwargs.get("data", {})
    if data.get("geometry") is not None or ("data" in kwargs and "boundaries" not in data):
        from nf2.train.config import normalize_config

        kwargs = normalize_config(kwargs)
    from nf2.extrapolate_series import run as _run_series

    return _run_series(*args, **kwargs)


def export_file(*args, **kwargs):
    """Export one NF2 result file to VTK, NPZ, HDF5, or FITS."""
    from nf2.export import export_file as _export_file

    return _export_file(*args, **kwargs)


def export_series(*args, **kwargs):
    """Export multiple NF2 result files matched by one or more glob patterns."""
    from nf2.export import export_series as _export_series

    return _export_series(*args, **kwargs)


def download_sharp_series(*args, **kwargs):
    """Download a SHARP data series through JSOC/DRMS."""
    from nf2.data.download import download_SHARP_series

    return download_SHARP_series(*args, **kwargs)


def download_hmi_full_disk(*args, **kwargs):
    """Download HMI full-disk vector data through JSOC/DRMS."""
    from nf2.data.download import download_hmi_full_disk as _download_hmi_full_disk

    return _download_hmi_full_disk(*args, **kwargs)


def download_hmi_sharp(*args, **kwargs):
    """Download HMI SHARP vector data through JSOC/DRMS."""
    from nf2.data.download import download_hmi_sharp as _download_hmi_sharp

    return _download_hmi_sharp(*args, **kwargs)


def download_hmi_synoptic(*args, **kwargs):
    """Download HMI synoptic vector maps through JSOC/DRMS."""
    from nf2.data.download import download_hmi_synoptic as _download_hmi_synoptic

    return _download_hmi_synoptic(*args, **kwargs)


def __getattr__(name):
    if name in {"CartesianOutput", "DisambiguationOutput", "HeightTransformOutput", "SphericalOutput"}:
        from nf2.evaluation import output

        return getattr(output, name)
    raise AttributeError(name)


__all__ = [
    "CartesianOutput",
    "DisambiguationOutput",
    "HeightTransformOutput",
    "SphericalOutput",
    "download_hmi_full_disk",
    "download_hmi_sharp",
    "download_hmi_synoptic",
    "download_sharp_series",
    "export_file",
    "export_series",
    "load",
    "run",
    "run_series",
]
