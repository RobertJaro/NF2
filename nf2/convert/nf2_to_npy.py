import argparse
import os.path

import numpy as np

from nf2.evaluation.output import CartesianOutput


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.npy')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, **kwargs)
    np.save(out_path, output['b'])


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target NPY file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path
    height_range = args.height_range

    convert(nf2_path, out_path, Mm_per_pixel, height_range)


if __name__ == '__main__':
    main()
