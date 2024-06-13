import argparse
import glob
import os.path

from nf2.convert import nf2_to_vtk, nf2_to_hdf5, nf2_to_npy, nf2_to_fits


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_dir', type=str, help='path to the source NF2 files')
    parser.add_argument('--out_dir', type=str, help='path to the target VTK directory', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)
    parser.add_argument('--type', type=str, help='type of the conversion (vtk, hdf5, npy, fits)', required=False,
                        default='vtk')

    args = parser.parse_args()
    nf2_paths = sorted(glob.glob(args.nf2_dir))

    Mm_per_pixel = args.Mm_per_pixel
    out_dir = args.out_dir if args.out_dir is not None else os.path.dirname(args.nf2_dir)
    height_range = args.height_range

    conversion_type = args.type
    if conversion_type == 'vtk':
        d_type = '.vtk'
        convert_f = nf2_to_vtk.convert
    elif conversion_type == 'hdf5':
        d_type = '.hdf5'
        convert_f = nf2_to_hdf5.convert
    elif conversion_type == 'npy':
        d_type = '.npy'
        convert_f = nf2_to_npy.convert
    elif conversion_type == 'fits':
        d_type = '.fits'
        convert_f = nf2_to_fits.convert
    else:
        raise ValueError(f'Unknown conversion type: {conversion_type}')

    for nf2_path in nf2_paths:
        out_file = os.path.join(out_dir, os.path.basename(nf2_path).replace('.nf2', d_type))
        convert_f(nf2_path=nf2_path, out_path=out_file,
                  Mm_per_pixel=Mm_per_pixel, height_range=height_range)


if __name__ == '__main__':
    main()
