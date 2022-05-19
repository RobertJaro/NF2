import argparse
import glob
import os

import torch

from nf2.evaluation.unpack import load_cube
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
parser.add_argument('vtk_path', type=str, help='path to the directory of output VTK files')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
nf2_path = args.nf2_path
vtk_path = args.vtk_path
strides = args.strides
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for i, f in enumerate(sorted(glob.glob(os.path.join(nf2_path, '*.nf2')))):
    b = load_cube(f, device, progress=True, strides=strides)
    save_vtk(b, os.path.join(vtk_path, 'series_%04d.vtk' % i), 'B')
