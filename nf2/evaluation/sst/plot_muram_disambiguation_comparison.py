import argparse

import numpy as np

from nf2.evaluation.sst.disambiguation_comparison import DisambiguationComparison


class MuramDisambiguationComparison(DisambiguationComparison):
    def __init__(self, multi_height_path, ambiguous_path, out_path, output_name, Mm_per_pixel, slice_path):
        super().__init__(
            nf2_paths=[multi_height_path, ambiguous_path],
            out_path=out_path,
            output_name=output_name,
            Mm_per_pixel=Mm_per_pixel,
            reference_name='MURaM',
            nf2_names=['Multi-height', 'Ambiguous'],
        )
        self.slice_path = slice_path

    def load_reference_field(self):
        from nf2.loader.muram import read_muram_slice

        sl, _n_var, _shape, _time = read_muram_slice(self.slice_path)
        bz = sl[5, :, :] * np.sqrt(4 * np.pi)
        bx = sl[6, :, :] * np.sqrt(4 * np.pi)
        by = sl[7, :, :] * np.sqrt(4 * np.pi)
        return np.stack([bx, by, bz], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot MURaM disambiguation comparison.')
    parser.add_argument('--multi-height-path', required=True)
    parser.add_argument('--ambiguous-path', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--Mm_per_pixel', type=float, required=True)
    parser.add_argument('--slice-path', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    MuramDisambiguationComparison(
        multi_height_path=args.multi_height_path,
        ambiguous_path=args.ambiguous_path,
        out_path=args.out_path,
        output_name=args.output_name,
        Mm_per_pixel=args.Mm_per_pixel,
        slice_path=args.slice_path,
    ).run()
