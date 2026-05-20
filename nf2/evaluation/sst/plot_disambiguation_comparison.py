import argparse

import numpy as np
from astropy.io import fits

from nf2.evaluation.sst.disambiguation_comparison import DisambiguationComparison


class SharpDisambiguationComparison(DisambiguationComparison):
    def __init__(
            self,
            nf2_path,
            ambiguous_path,
            out_path,
            output_name,
            Mm_per_pixel,
            field_path,
            inclination_path,
            azimuth_path,
            reference_slice):
        super().__init__(
            nf2_paths=[nf2_path, ambiguous_path],
            out_path=out_path,
            output_name=output_name,
            Mm_per_pixel=Mm_per_pixel,
            reference_name='SHARP',
            nf2_names=['NF2', 'NF2 ambiguous'],
        )
        self.field_path = field_path
        self.inclination_path = inclination_path
        self.azimuth_path = azimuth_path
        self.reference_slice = reference_slice

    def load_reference_field(self):
        b_fld = fits.getdata(self.field_path)
        b_inc = fits.getdata(self.inclination_path)
        b_azi = fits.getdata(self.azimuth_path)

        b_fld = np.flip(b_fld, axis=(0, 1))
        b_inc = np.flip(b_inc, axis=(0, 1))
        b_azi = np.flip(b_azi, axis=(0, 1))

        bx = b_fld * np.sin(np.deg2rad(b_inc)) * np.sin(np.deg2rad(b_azi))
        by = -b_fld * np.sin(np.deg2rad(b_inc)) * np.cos(np.deg2rad(b_azi))
        bz = b_fld * np.cos(np.deg2rad(b_inc))
        b = np.stack([bx, by, bz]).T

        y0, y1, x0, x1 = self.reference_slice
        return b[y0:y1, x0:x1, :]


def parse_args():
    parser = argparse.ArgumentParser(description='Plot SHARP disambiguation comparison.')
    parser.add_argument('--nf2-path', required=True)
    parser.add_argument('--ambiguous-path', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--Mm_per_pixel', type=float, required=True)
    parser.add_argument('--field-path', required=True)
    parser.add_argument('--inclination-path', required=True)
    parser.add_argument('--azimuth-path', required=True)
    parser.add_argument('--reference-slice', nargs=4, type=int, required=True, metavar=('Y0', 'Y1', 'X0', 'X1'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    SharpDisambiguationComparison(
        nf2_path=args.nf2_path,
        ambiguous_path=args.ambiguous_path,
        out_path=args.out_path,
        output_name=args.output_name,
        Mm_per_pixel=args.Mm_per_pixel,
        field_path=args.field_path,
        inclination_path=args.inclination_path,
        azimuth_path=args.azimuth_path,
        reference_slice=args.reference_slice,
    ).run()
