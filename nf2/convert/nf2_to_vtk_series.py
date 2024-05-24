import argparse
import glob
import os.path

from nf2.convert import nf2_to_vtk


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 files')
    parser.add_argument('--vtk_path', type=str, help='path to the target VTK directory', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_paths = sorted(glob.glob(args.nf2_path))

    Mm_per_pixel = args.Mm_per_pixel
    vtk_path = args.vtk_path if args.vtk_path is not None else os.path.dirname(args.nf2_path)
    height_range = args.height_range

    for nf2_path in nf2_paths:
        vtk_file = os.path.join(vtk_path, os.path.basename(nf2_path).replace('.nf2', '.vtk'))
        nf2_to_vtk.convert(nf2_path, vtk_file, Mm_per_pixel, height_range)


if __name__ == '__main__':
    main()
