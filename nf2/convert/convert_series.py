import argparse
import glob
import os.path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_dir', type=str, help='path to the source NF2 files', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, help='path to the target VTK directory', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)
    parser.add_argument('--metrics', type=str, nargs='*', help='metrics to be computed', required=False, default=['j'])
    parser.add_argument('--type', type=str, help='type of the conversion (vtk, hdf5, npy, fits)', required=False,
                        default='vtk')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files', required=False,
                        default=False)

    args = parser.parse_args()

    nf2_dir = args.nf2_dir if isinstance(args.nf2_dir, list) else [args.nf2_dir]
    nf2_paths = [sorted(glob.glob(f)) for f in nf2_dir]
    nf2_paths = [f for files in nf2_paths for f in files]  # flatten list

    Mm_per_pixel = args.Mm_per_pixel
    out_dir = args.out_dir if args.out_dir is not None else os.path.dirname(nf2_dir[0])
    os.makedirs(out_dir, exist_ok=True)
    height_range = args.height_range
    metrics = args.metrics

    conversion_type = args.type
    if conversion_type == 'vtk':
        d_type = '.vtk'
        from nf2.convert import nf2_to_vtk
        convert_f = nf2_to_vtk.convert
    elif conversion_type == 'hdf5':
        d_type = '.hdf5'
        from nf2.convert import nf2_to_hdf5
        convert_f = nf2_to_hdf5.convert
    elif conversion_type == 'npy':
        d_type = '.npy'
        from nf2.convert import nf2_to_npy
        convert_f = nf2_to_npy.convert
    elif conversion_type == 'fits':
        d_type = '.fits'
        from nf2.convert import nf2_to_fits
        convert_f = nf2_to_fits.convert
    else:
        raise ValueError(f'Unknown conversion type: {conversion_type}')

    for nf2_path in tqdm(nf2_paths, desc='Converting', unit='file'):
        out_file = os.path.join(out_dir, os.path.basename(nf2_path).replace('.nf2', d_type))
        if os.path.exists(out_file) and not args.overwrite:
            print(f'File exists: {out_file}')
            continue
        convert_f(nf2_path=nf2_path, out_path=out_file,
                  Mm_per_pixel=Mm_per_pixel, height_range=height_range, metrics=metrics,
                  progress=False)


if __name__ == '__main__':
    main()
