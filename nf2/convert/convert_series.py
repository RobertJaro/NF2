import argparse
import os
import os.path

from nf2.export.core import export_series


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 files as a series.')
    parser.add_argument('--nf2_dir', type=str, help='path to the source NF2 files', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, help='path to the target directory', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)
    parser.add_argument('--metrics', type=str, nargs='*', help='metrics to be computed', required=False, default=['j'])
    parser.add_argument('--type', type=str, help='type of the conversion (vtk, hdf5, npz, fits, binary)', required=False,
                        default='vtk')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files', required=False,
                        default=False)

    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else os.path.dirname(args.nf2_dir[0])
    os.makedirs(out_dir, exist_ok=True)

    export_series(
        args.nf2_dir,
        args.type,
        out_dir=out_dir,
        overwrite=args.overwrite,
        Mm_per_pixel=args.Mm_per_pixel,
        height_range=args.height_range,
        metrics=args.metrics,
        progress=False,
    )


if __name__ == '__main__':
    main()
