import argparse

import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.vtk import save_vtk
from nf2.loader.muram import MURaMSnapshot

from astropy import units as u

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--source_path', type=str, help='path to the MURaM simulation.')
parser.add_argument('--iteration', type=int, help='iteration of the snapshot.')
parser.add_argument('--vtk_path', type=str, help='path to the target VTK file', required=False, default=None)
parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False, default=0.72)
args = parser.parse_args()

Mm_per_pixel = args.Mm_per_pixel * u.Mm / u.pix

snapshot = MURaMSnapshot(args.source_path, args.iteration)
snapshot_ds = (0.192 * u.Mm / u.pix, 0.192 * u.Mm / u.pix, 0.192 / 2 * u.Mm / u.pix)


tau_cube = snapshot.tau
pix_height = np.argmin(np.abs(tau_cube - 1), axis=2) * u.pix
base_height_pix = pix_height.min()

ds = snapshot_ds
b = snapshot.B
b = block_reduce(b, (4, 4, 8, 1), np.mean)
b = b[:, :, int(base_height_pix.to_value(u.pix) / 8):]

save_vtk(args.vtk_path, vectors={'B': b}, Mm_per_pix=(snapshot_ds[0] * 4).to_value(u.Mm / u.pix))
