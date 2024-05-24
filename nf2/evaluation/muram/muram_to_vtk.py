import argparse

import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.metric import curl
from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.vtk import save_vtk
from nf2.loader.muram import MURaMSnapshot

from astropy import units as u

from astropy import constants as const

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--source_path', type=str, help='path to the MURaM simulation.')
parser.add_argument('--iteration', type=int, help='iteration of the snapshot.')
parser.add_argument('--vtk_path', type=str, help='path to the target VTK file', required=False, default=None)
parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (needs to be a multiple of 0.192)', required=False, default=0.192 * 2)
args = parser.parse_args()

Mm_per_pixel = args.Mm_per_pixel * u.Mm / u.pix

snapshot = MURaMSnapshot(args.source_path, args.iteration)
muram_cube = snapshot.load_base(args.Mm_per_pixel * u.Mm / u.pix, base_height=0)
b = muram_cube['B']
j = curl(b) * u.G / (args.Mm_per_pixel * u.Mm) * const.c / (4 * np.pi) # Mm_per_pixel
j = j.to(u.G / u.s)

azimuth = np.arctan2(b[..., 1], b[..., 0])

save_vtk(args.vtk_path, vectors={'B': b, 'J': j.value},
         scalars={'tau': muram_cube['tau'], 'azimuth': azimuth}, Mm_per_pix=args.Mm_per_pixel)
