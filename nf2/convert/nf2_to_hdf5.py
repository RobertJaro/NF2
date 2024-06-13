import argparse
import os.path

import h5py

from nf2.evaluation.output import CartesianOutput, current_density, los_trv_azi, b_nabla_bz


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.hdf5')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, progress=True,
                               metrics={'j': current_density})

    f = h5py.File(out_path, 'w')
    f.create_dataset('B', data=output['b'], dtype='f4')
    f.create_dataset('J', data=output['j'], dtype='f4')
    f.attrs['Mm_per_pix'] = output['Mm_per_pixel']
    f.attrs['data'] = nf2_out.data_config
    f.attrs['wcs'] = nf2_out.wcs
    f.attrs['time'] = nf2_out.time
    f.close()


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--hdf5_path', type=str, help='path to the target HDF5 file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    hdf5_path = args.hdf5_path
    height_range = args.height_range

    convert(nf2_path, hdf5_path, Mm_per_pixel, height_range)


if __name__ == '__main__':
    main()
