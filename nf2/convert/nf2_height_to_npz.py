import argparse
import os.path
import pickle

from astropy import units as u

from nf2.evaluation.output import HeightTransformOutput


def _add_magnetic_field(output, nf2_out):
    for entry in output:
        coords = entry['coords'].to_value(u.Mm) / nf2_out.Mm_per_ds
        b = nf2_out.load_coords(coords, compute_jacobian=False)['b']
        if b.ndim == 4 and b.shape[2] == 1:
            b = b[:, :, 0]
        entry['b'] = b
    return output


def convert(nf2_path, out_path=None, Mm_per_pixel=None, include_b=True, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.npy')

    nf2_out = HeightTransformOutput(nf2_path)
    output = nf2_out.load_height_mapping(Mm_per_pixel=Mm_per_pixel)
    if include_b:
        output = _add_magnetic_field(output, nf2_out)

    # save outputs
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target PKL file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--no_b', action='store_true', help='Do not save magnetic field vectors at the height layers.')

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path

    if out_path is None:
        out_path = os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.npy')

    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    convert(nf2_path, out_path, Mm_per_pixel, include_b=not args.no_b)


if __name__ == '__main__':
    main()
