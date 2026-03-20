import argparse

from astropy import units as u

from nf2.evaluation.output_metrics import current_density, alpha
from nf2.export.core import export_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert spherical NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)

    parser.add_argument('--radius_range', nargs='+', type=float, default=(0.999, 1.3), required=False)
    parser.add_argument('--latitude_range', nargs='+', type=float, default=(-90, 90), required=False)
    parser.add_argument('--longitude_range', nargs='+', type=float, default=(0, 360), required=False)
    parser.add_argument('--radians', action='store_true', help='latitude and longitude in radians', required=False,
                        default=False)
    parser.add_argument('--pixels_per_solRad', type=int, default=64, required=False)

    args = parser.parse_args()

    radius_range = tuple(args.radius_range)
    if args.radians:
        latitude_range = tuple(args.latitude_range) * u.rad
        longitude_range = tuple(args.longitude_range) * u.rad
    else:
        latitude_range = tuple(args.latitude_range) * u.deg
        longitude_range = tuple(args.longitude_range) * u.deg

    export_checkpoint(
        args.nf2_path,
        "vtk",
        out_path=args.out_path,
        radius_range=radius_range,
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        pixels_per_solRad=args.pixels_per_solRad,
        progress=True,
        metrics={"j": current_density, "alpha": alpha},
    )
