import argparse

import torch

from nf2.evaluation.unpack import load_cube, load_height_cube
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('vtk_path', type=str, help='path to the target VTK file')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
nf2_path = args.nf2_path
strides = args.strides
vtk_path = args.vtk_path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

b = load_cube(nf2_path, device, progress=True, strides=strides)
tau = load_height_cube(nf2_path, device=device, progress=True, strides=strides)

save_vtk(b, vtk_path, 'B', scalar=tau, scalar_name='tau',
         Mm_per_pix=torch.load(nf2_path)['Mm_per_pixel'])

save_vtk(b, vtk_path, 'B',   Mm_per_pix=torch.load(nf2_path)['Mm_per_pixel'])