"""Unified export command for NF2 result files."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def _geometry(nf2_path: str) -> str | None:
    import torch

    state = torch.load(nf2_path, map_location="cpu", weights_only=False)
    return state.get("data", {}).get("type")


def _normalize_format(fmt: str) -> str:
    fmt = fmt.lower().replace("-", "_")
    if fmt in {"height", "height_npz", "heights", "heights_npz"}:
        return "height"
    return fmt


def export_file(
    nf2_path: str,
    out_path: str | None = None,
    *,
    fmt: str = "vtk",
    Mm_per_pixel: float | None = None,
    height_range: list[float] | None = None,
    metrics: list[str] | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    progress: bool = True,
):
    """Export one NF2 file to a supported exchange format.

    Parameters
    ----------
    nf2_path:
        Path to an ``extrapolation_result.nf2`` checkpoint.
    out_path:
        Optional output path. Converter defaults are used when omitted.
    fmt:
        Export format: ``vtk``, ``npz``, ``hdf5``/``h5``, ``fits``, or
        ``height`` for multi-height surface mappings.
    Mm_per_pixel, height_range, x_range, y_range:
        Cartesian sampling controls in megameters.
    metrics:
        Derived quantities to include, such as ``j``, ``alpha``, or
        ``free_energy_fft``.
    progress:
        Show converter progress where supported.
    """
    fmt = _normalize_format(fmt)
    metrics = metrics if metrics is not None else ["j"]

    if fmt == "vtk":
        if _geometry(nf2_path) == "spherical":
            from nf2.convert.nf2_to_vtk_spherical import convert

            return convert(
                nf2_path=nf2_path,
                out_path=out_path,
                metrics=metrics,
                progress=progress,
            )

        from nf2.convert.nf2_to_vtk import convert

        return convert(
            nf2_path=nf2_path,
            out_path=out_path,
            Mm_per_pixel=Mm_per_pixel,
            height_range=height_range,
            metrics=metrics,
            x_range=x_range,
            y_range=y_range,
            progress=progress,
        )
    if fmt in {"npz", "npy"}:
        from nf2.convert.nf2_to_npz import convert

        return convert(
            nf2_path=nf2_path,
            out_path=out_path,
            Mm_per_pixel=Mm_per_pixel,
            height_range=height_range,
            metrics=metrics,
            x_range=x_range,
            y_range=y_range,
            progress=progress,
        )
    if fmt == "height":
        from nf2.convert.nf2_height_to_npz import convert

        return convert(
            nf2_path=nf2_path,
            out_path=out_path,
            Mm_per_pixel=Mm_per_pixel,
            progress=progress,
        )
    if fmt in {"hdf5", "h5"}:
        from nf2.convert.nf2_to_hdf5 import convert

        return convert(
            nf2_path=nf2_path,
            out_path=out_path,
            Mm_per_pixel=Mm_per_pixel,
            height_range=height_range,
            metrics=metrics,
            progress=progress,
        )
    if fmt == "fits":
        from nf2.convert.nf2_to_fits import convert

        return convert(
            nf2_path=nf2_path,
            out_path=out_path,
            Mm_per_pixel=Mm_per_pixel,
            height_range=height_range,
            metrics=metrics,
            progress=progress,
        )
    raise ValueError(f"Unsupported export format: {fmt}")


def export_series(
    patterns: list[str],
    out_dir: str,
    *,
    fmt: str = "vtk",
    overwrite: bool = False,
    **kwargs,
):
    """Export all NF2 files matched by one or more glob patterns.

    Existing files are skipped unless ``overwrite`` is true.
    """
    fmt = _normalize_format(fmt)
    suffix = {
        "vtk": ".vtk",
        "npz": ".npz",
        "npy": ".npz",
        "height": ".height.npz",
        "hdf5": ".hdf5",
        "h5": ".hdf5",
        "fits": ".fits",
    }[fmt]
    nf2_paths = [path for pattern in patterns for path in sorted(glob.glob(pattern))]
    os.makedirs(out_dir, exist_ok=True)

    outputs = []
    for nf2_path in nf2_paths:
        out_path = os.path.join(out_dir, Path(nf2_path).with_suffix(suffix).name)
        if os.path.exists(out_path) and not overwrite:
            continue
        outputs.append(export_file(nf2_path, out_path, fmt=fmt, progress=False, **kwargs))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Export NF2 extrapolation results.")
    parser.add_argument("nf2_path", nargs="+", help="NF2 file path or glob pattern")
    parser.add_argument(
        "--format",
        choices=["vtk", "npz", "hdf5", "h5", "fits", "height", "height-npz", "height_npz"],
        default="vtk",
    )
    parser.add_argument("--out", help="Output file for a single input")
    parser.add_argument("--out-dir", help="Output directory for multiple inputs")
    parser.add_argument("--Mm_per_pixel", type=float, default=None)
    parser.add_argument("--height_range", type=float, nargs=2, default=None)
    parser.add_argument("--x_range", type=float, nargs=2, default=None)
    parser.add_argument("--y_range", type=float, nargs=2, default=None)
    parser.add_argument("--metrics", nargs="*", default=["j"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    matched = [path for pattern in args.nf2_path for path in sorted(glob.glob(pattern))]
    if len(matched) == 1 and args.out_dir is None:
        export_file(
            matched[0],
            args.out,
            fmt=args.format,
            Mm_per_pixel=args.Mm_per_pixel,
            height_range=args.height_range,
            metrics=args.metrics,
            x_range=args.x_range,
            y_range=args.y_range,
        )
        return

    out_dir = args.out_dir if args.out_dir is not None else os.getcwd()
    export_series(
        args.nf2_path,
        out_dir,
        fmt=args.format,
        overwrite=args.overwrite,
        Mm_per_pixel=args.Mm_per_pixel,
        height_range=args.height_range,
        metrics=args.metrics,
        x_range=args.x_range,
        y_range=args.y_range,
    )


if __name__ == "__main__":
    main()
