import argparse

import torch

from nf2.evaluation.unpack import load_cube
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('vtk_path', type=str, help='path to the target VTK file')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
b = load_cube(args.nf2_path, device, progress=True, strides=args.strides)
save_vtk(b, args.vtk_path, 'B')