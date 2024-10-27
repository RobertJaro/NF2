import argparse
import os

from astropy.io import fits

from nf2.evaluation.output import CartesianOutput


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.hdf5')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, **kwargs)

    b = output['b']
    j = output['j']

    header = {'Mm_per_pix': output['Mm_per_pixel'],
              'data': nf2_out.data_config,
              'wcs': nf2_out.wcs,
              'DATE_OBS': nf2_out.time}

    b_hdu = fits.PrimaryHDU(b, header=header)
    j_hdu = fits.FitsHDU(j)
    hdul = fits.HDUList([b_hdu, j_hdu])
    hdul.writeto(out_path)


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to fits.')
    parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target numpy file', required=False, default=None)
    parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

    args = parser.parse_args()
    nf2_path = args.nf2_path
    strides = args.strides
    out_path = args.out_path

    convert(nf2_path, out_path, strides)


if __name__ == '__main__':
    main()
