from __future__ import annotations

import glob
import os

from nf2.output.checkpoint import NF2Checkpoint
from nf2.export.writers import WRITERS


def _default_path(checkpoint_path, export_format):
    if export_format == "binary":
        return os.path.splitext(checkpoint_path)[0]
    extension = ".npz" if export_format == "npz" else f".{export_format}"
    return os.path.splitext(checkpoint_path)[0] + extension


def _build_output_adapter(checkpoint_path, device=None):
    checkpoint = NF2Checkpoint(checkpoint_path, device=device)
    if checkpoint.geometry == "cartesian":
        from nf2.output.adapters import CartesianOutputAdapter

        return CartesianOutputAdapter(checkpoint_path, device=device)
    if checkpoint.geometry == "spherical":
        from nf2.output.adapters import SphericalOutputAdapter

        return SphericalOutputAdapter(checkpoint_path, device=device)
    raise ValueError(f"Unsupported checkpoint geometry: {checkpoint.geometry}")


def export_checkpoint(checkpoint_path, export_format, out_path=None, device=None, **sample_kwargs):
    if export_format not in WRITERS:
        raise ValueError(f"Unsupported export format: {export_format}")
    adapter = _build_output_adapter(checkpoint_path, device=device)
    payload = adapter.sample(**sample_kwargs)
    out_path = _default_path(checkpoint_path, export_format) if out_path is None else out_path
    WRITERS[export_format](out_path, payload)
    return out_path


def export_series(patterns, export_format, out_dir=None, overwrite=False, device=None, **sample_kwargs):
    patterns = patterns if isinstance(patterns, list) else [patterns]
    checkpoint_paths = [path for pattern in patterns for path in sorted(glob.glob(pattern))]
    exported = []
    for checkpoint_path in checkpoint_paths:
        base_name = os.path.basename(os.path.splitext(checkpoint_path)[0])
        if out_dir is None:
            out_path = None
        elif export_format == "binary":
            out_path = os.path.join(out_dir, base_name)
        else:
            ext = ".npz" if export_format == "npz" else f".{export_format}"
            out_path = os.path.join(out_dir, base_name + ext)
        target = _default_path(checkpoint_path, export_format) if out_path is None else out_path
        if os.path.exists(target) and not overwrite:
            continue
        exported.append(
            export_checkpoint(checkpoint_path, export_format, out_path=out_path, device=device, **sample_kwargs)
        )
    return exported
