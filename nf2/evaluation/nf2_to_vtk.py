import argparse

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--vtk_path', type=str, help='path to the target VTK file', required=False, default=None)
parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False, default=None)
parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

args = parser.parse_args()
nf2_path = args.nf2_path
vtk_path = args.vtk_path if args.vtk_path is not None else nf2_path.replace('.nf2', '.vtk')

n2_out = CartesianOutput(nf2_path)
output = n2_out.load_cube(Mm_per_pixel=args.Mm_per_pixel, height_range=args.height_range, progress=True)

save_vtk(vtk_path, vectors={'B': output['B'], 'J': output['J']}, Mm_per_pix=output['Mm_per_pixel'])
