import argparse
import os.path

from nf2.evaluation.output import CartesianOutput, current_density, b_nabla_bz, magnetic_helicity
from nf2.evaluation.vtk import save_vtk


def convert(nf2_path, vtk_path=None, Mm_per_pixel=None, height_range=None):
    vtk_path = vtk_path if vtk_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.vtk')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, progress=True,
                               metrics={'j': current_density, 'b_nabla_bz': b_nabla_bz,
                                        'magnetic_helicity': magnetic_helicity})

    save_vtk(vtk_path, vectors={'B': output['b'], 'J': output['j']},
             scalars={'b_nabla_bz': output['b_nabla_bz'],
                      'magnetic_helicity': output['magnetic_helicity']}, Mm_per_pix=output['Mm_per_pixel'])


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--vtk_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    vtk_path = args.vtk_path
    height_range = args.height_range

    convert(nf2_path, vtk_path, Mm_per_pixel, height_range)


if __name__ == '__main__':
    main()
