import argparse

from nf2.export.core import export_checkpoint


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, **kwargs):
    return export_checkpoint(
        nf2_path,
        "binary",
        out_path=out_path,
        Mm_per_pixel=Mm_per_pixel,
        height_range=height_range,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to binary.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target directory', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    convert(args.nf2_path, args.out_path, args.Mm_per_pixel, args.height_range, progress=True)


if __name__ == '__main__':
    main()
