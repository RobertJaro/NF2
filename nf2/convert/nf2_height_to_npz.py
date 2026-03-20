import argparse
import os.path
import pickle

import numpy as np

from nf2.evaluation.output import HeightTransformOutput


def convert(nf2_path, out_path=None, Mm_per_pixel=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.pkl')

    nf2_out = HeightTransformOutput(nf2_path)
    output = nf2_out.load_height_mapping(Mm_per_pixel=Mm_per_pixel)

    # save outputs
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 height mappings to PKL.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target PKL file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path

    if out_path is not None:
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    convert(nf2_path, out_path, Mm_per_pixel)


if __name__ == '__main__':
    main()
