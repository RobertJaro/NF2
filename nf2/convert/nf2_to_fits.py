import argparse

import torch
from astropy.io import fits

from nf2.evaluation.unpack import load_cube


def convert(nf2_path, fits_path, strides):
    fits_path = fits_path if fits_path is not None else nf2_path.replace('.nf2', '.fits')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = load_cube(nf2_path, device, progress=True, strides=strides)
    hdu = fits.PrimaryHDU(b)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_path)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to fits.')
    parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--fits_path', type=str, help='path to the target numpy file', required=False, default=None)
    parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

    args = parser.parse_args()
    nf2_path = args.nf2_path
    strides = args.strides
    fits_path = args.fits_path

    convert(nf2_path, fits_path, strides)


if __name__ == '__main__':
    main()
