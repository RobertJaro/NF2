import argparse
import os.path

import numpy as np

from nf2.evaluation.output import HeightTransformOutput


def _as_array(value):
    if hasattr(value, "to_value"):
        value = value.to_value()
    return np.asarray(value)


def convert(nf2_path, out_path=None, Mm_per_pixel=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '_height.npz')

    nf2_out = HeightTransformOutput(nf2_path)
    output = nf2_out.load_height_mapping(Mm_per_pixel=Mm_per_pixel, **kwargs)

    save_dict = {'n_mappings': np.array(len(output), dtype=np.int32)}
    for i, entry in enumerate(output):
        prefix = f'mapping_{i}_'
        save_dict[prefix + 'height'] = _as_array(entry['height'])
        save_dict[prefix + 'coords'] = _as_array(entry['coords'])
        save_dict[prefix + 'original_coords'] = _as_array(entry['original_coords'])
        save_dict[prefix + 'Mm_per_pixel'] = np.asarray(entry['Mm_per_pixel'])
    np.savez(out_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 height mapping to NPZ.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target NPZ file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path

    dirname = os.path.dirname(out_path) if out_path is not None else None
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    convert(nf2_path, out_path, Mm_per_pixel)


if __name__ == '__main__':
    main()
