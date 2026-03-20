import argparse

from nf2.export.core import export_checkpoint


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, metrics=None, x_range=None, y_range=None, **kwargs):
    return export_checkpoint(
        nf2_path,
        "vtk",
        out_path=out_path,
        Mm_per_pixel=Mm_per_pixel,
        height_range=height_range,
        metrics=metrics,
        x_range=x_range,
        y_range=y_range,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)
    parser.add_argument('--x_range', type=float, nargs=2, help='x range in Mm', required=False, default=None)
    parser.add_argument('--y_range', type=float, nargs=2, help='y range in Mm', required=False, default=None)
    parser.add_argument('--metrics', type=str, nargs='*', help='metrics to be computed', required=False, default=['j'])

    args = parser.parse_args()
    convert(args.nf2_path, args.out_path, args.Mm_per_pixel, args.height_range, metrics=args.metrics, progress=True,
            x_range=args.x_range, y_range=args.y_range)


if __name__ == '__main__':
    main()
