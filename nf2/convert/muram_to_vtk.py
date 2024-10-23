import argparse

import numpy as np
from astropy import constants as const
from astropy import units as u

from nf2.evaluation.metric import curl
from nf2.evaluation.vtk import save_vtk
from nf2.loader.muram import MURaMSnapshot


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--source_path', type=str, help='path to the MURaM simulation.')
    parser.add_argument('--iteration', type=int, help='iteration of the snapshot.')
    parser.add_argument('--vtk_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (needs to be a multiple of 0.192)',
                        required=False, default=0.192 * 4)
    parser.add_argument('--height', type=float, help='height of the snapshot', required=False, default=None)
    args = parser.parse_args()

    snapshot = MURaMSnapshot(args.source_path, args.iteration)
    muram_cube = snapshot.load_cube(args.Mm_per_pixel * u.Mm / u.pix, target_tau=1.0, height=args.height * u.Mm)
    b = muram_cube['B']
    j = curl(b) * u.G / (args.Mm_per_pixel * u.Mm) * const.c / (4 * np.pi)  # Mm_per_pixel
    j = j.to(u.G / u.s)

    save_vtk(args.vtk_path, vectors={'b': b, 'j': j.value},
             scalars={'tau': muram_cube['tau']}, Mm_per_pix=args.Mm_per_pixel)


if __name__ == '__main__':
    main()
