import argparse
import glob
import gzip
import os
import shutil

import torch
from tqdm import tqdm

from nf2.evaluation.unpack import load_cube, save_fits

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
parser.add_argument('fits_path', type=str, help='path to the directory of output FITS files')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)
parser.add_argument('--height', type=int, help='maximum height of the sampled volume in pixels (default none).',
                    required=False, default=None)
parser.add_argument('--gz_files', action='store_true')

args = parser.parse_args()
nf2_path = args.nf2_path
fits_path = args.fits_path
strides = args.strides
height = args.height
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

os.makedirs(fits_path, exist_ok=True)


def _zip_paths(map_paths):
    for map_path in map_paths:
        with open(map_path, 'rb') as f_in, gzip.open(map_path + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(map_path)


for f in tqdm(sorted(glob.glob(os.path.join(nf2_path, '**', '*.nf2')))):
    b = load_cube(f, device, progress=False, strides=strides, z=height)
    for z in range(height):
        date_str = f.split('/')[-2][:15].replace('_', 'T')
        meta = {'DATE-OBS': date_str, 'HEIGHT': z * 360e-3}
        map_paths = save_fits(b[:, :, z], fits_path, '%s_%02d' % (date_str, z), meta)
        if args.gz_files:
            _zip_paths(map_paths)

shutil.make_archive(fits_path, 'zip', fits_path)
