import argparse
import os.path

import numpy as np
from astropy import constants
from astropy import units as u

from nf2.evaluation.metric import b_nabla_bz, curl
from nf2.evaluation.output_metrics import free_energy, squashing_factor
from nf2.loader.muram import MURaMSnapshot


def convert(muram_source_path, muram_iteration, out_path, Mm_per_pixel=0.192, height=100 * u.Mm, **kwargs):
    mfr_muram_snapshot = MURaMSnapshot(muram_source_path, iteration=muram_iteration)
    mfr_muram_cube = mfr_muram_snapshot.load_cube(resolution=Mm_per_pixel * u.Mm / u.pix, height=height,
                                                  method='min')
    b = mfr_muram_cube['B']
    tau = mfr_muram_cube['tau']
    muram_bnbz = b_nabla_bz(b) / Mm_per_pixel
    j = (curl(b) / Mm_per_pixel * u.G / u.Mm) * constants.c / (4 * np.pi)
    j = j.to_value(u.G / u.s)

    free_energy_out = free_energy(b * u.G)  # erg cm^-3

    # save outputs
    save_dict = {'b': b, 'Mm_per_pixel': Mm_per_pixel, 'j': j, 'b_nabla_bz': muram_bnbz, 'tau': tau, **free_energy_out}

    np.savez(out_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--muram_source_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--muram_iteration', type=int, help='MURaM iteration number')
    parser.add_argument('--out_path', type=str, help='path to the target NPY file', required=True)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.192 default)', required=False,
                        default=0.192)
    parser.add_argument('--height', type=float, help='height in Mm', required=False, default=50)
    args = parser.parse_args()
    muram_source_path = args.muram_source_path
    muram_iteration = args.muram_iteration

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path
    height = args.height

    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    convert(muram_source_path, muram_iteration, out_path, Mm_per_pixel,
            height=height * u.Mm)


if __name__ == '__main__':
    main()
