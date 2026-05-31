import argparse
import os
import re

from astropy.io import fits

from nf2.evaluation.output import CartesianOutput


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.fits')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, **kwargs)

    b = output['b']
    metrics = output.get('metrics', {})

    header = fits.Header()
    header['MM_P_PIX'] = float(output['Mm_per_pixel'])
    if nf2_out.time is not None:
        header['DATE-OBS'] = nf2_out.time.isoformat('T', timespec='seconds')
    if nf2_out.data_config is not None:
        header['NF2TYPE'] = str(nf2_out.data_config.get('type', 'unknown'))

    b_hdu = fits.PrimaryHDU(b, header=header)
    hdus = [b_hdu]
    for name, values in metrics.items():
        hdu_name = re.sub(r'[^A-Za-z0-9_]', '_', name).upper()
        hdus.append(fits.ImageHDU(values, name=hdu_name))
    hdul = fits.HDUList(hdus)
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
