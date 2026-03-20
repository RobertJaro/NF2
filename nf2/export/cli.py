import argparse

from astropy import units as u

from nf2.evaluation.output_metrics import alpha, current_density
from nf2.export.core import export_checkpoint, export_series


def _maybe_angle_range(values, radians=False):
    if values is None:
        return None
    return tuple(values) * (u.rad if radians else u.deg)


def main():
    parser = argparse.ArgumentParser(description="Unified NF2 export CLI.")
    parser.add_argument("--checkpoint", type=str, nargs="*", help="checkpoint path or glob")
    parser.add_argument("--series", action="store_true", help="treat checkpoint arguments as a series/glob")
    parser.add_argument("--format", type=str, required=True, choices=["vtk", "hdf5", "fits", "npz", "binary"])
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument("--Mm_per_pixel", type=float, default=None)
    parser.add_argument("--height_range", type=float, nargs=2, default=None)
    parser.add_argument("--x_range", type=float, nargs=2, default=None)
    parser.add_argument("--y_range", type=float, nargs=2, default=None)
    parser.add_argument("--metrics", type=str, nargs="*", default=["j"])

    parser.add_argument("--radius_range", type=float, nargs=2, default=None)
    parser.add_argument("--latitude_range", type=float, nargs=2, default=None)
    parser.add_argument("--longitude_range", type=float, nargs=2, default=None)
    parser.add_argument("--radians", action="store_true", default=False)
    parser.add_argument("--pixels_per_solRad", type=int, default=None)

    args = parser.parse_args()
    if not args.checkpoint:
        raise ValueError("Provide at least one --checkpoint path or glob")

    metrics = args.metrics
    if args.radius_range is not None:
        metric_map = {"j": current_density, "alpha": alpha}
        metrics = {name: metric_map[name] for name in args.metrics if name in metric_map}

    common_kwargs = {
        "Mm_per_pixel": args.Mm_per_pixel,
        "height_range": args.height_range,
        "x_range": args.x_range,
        "y_range": args.y_range,
        "metrics": metrics,
        "radius_range": args.radius_range,
        "latitude_range": _maybe_angle_range(args.latitude_range, radians=args.radians),
        "longitude_range": _maybe_angle_range(args.longitude_range, radians=args.radians),
        "pixels_per_solRad": args.pixels_per_solRad,
        "progress": True,
    }
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

    if args.series:
        export_series(args.checkpoint, args.format, out_dir=args.out_dir, overwrite=args.overwrite, **common_kwargs)
    else:
        if len(args.checkpoint) != 1:
            raise ValueError("Single export expects exactly one --checkpoint")
        export_checkpoint(args.checkpoint[0], args.format, out_path=args.out_path, **common_kwargs)


if __name__ == "__main__":
    main()
