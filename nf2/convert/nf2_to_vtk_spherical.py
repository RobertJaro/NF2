import argparse
import glob
import os
import threading

import numpy as np
from astropy import units as u

from nf2.evaluation.metric import normalized_divergence
from nf2.evaluation.output import SphericalOutput, current_density, twist
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files', required=False, default=None)
parser.add_argument('--temporal_strides', type=int, default=1, required=False,
                    help='select the time steps (default=1; all files).')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')

parser.add_argument('--radius_range', nargs='+', type=float, default=(0.999, 1.3), required=False)
parser.add_argument('--latitude_range', nargs='+', type=float, default=(0, 180), required=False)
parser.add_argument('--longitude_range', nargs='+', type=float, default=(0, 360), required=False)
parser.add_argument('--radians', action='store_true', help='latitude and longitude in radians', required=False,
                    default=False)
parser.add_argument('--pixels_per_solRad', type=int, default=64, required=False)

args = parser.parse_args()
nf2_path = args.nf2_path
out_path = args.out_path if args.out_path is not None else os.path.dirname(nf2_path)

os.makedirs(out_path, exist_ok=True)

radius_range = tuple(args.radius_range) * u.solRad
if args.radians:
    latitude_range = tuple(args.latitude_range) * u.rad
    longitude_range = tuple(args.longitude_range) * u.rad
else:
    latitude_range = tuple(args.latitude_range) * u.deg
    longitude_range = tuple(args.longitude_range) * u.deg
pixels_per_solRad = args.pixels_per_solRad * u.pix / u.solRad

assert len(radius_range) == 2, 'radius_range must be a tuple of length 2'
assert len(latitude_range) == 2, 'latitude_range must be a tuple of length 2'
assert len(longitude_range) == 2, 'longitude_range must be a tuple of length 2'

files = sorted(glob.glob(nf2_path))[::args.temporal_strides]  # if os.path.isdir(nf2_path) else [nf2_path]

save_threads = []
for i, f in enumerate(files):

    vtk_path = os.path.join(out_path, os.path.basename(f).replace('.nf2', '.vtk'))
    if os.path.exists(vtk_path) and not args.overwrite:
        print(f'Skipping existing file {f}')
        continue
    print('Processing file {} of {}'.format(i + 1, len(files)))

    output = SphericalOutput(f)
    result = output.load(radius_range, latitude_range, longitude_range, pixels_per_solRad, progress=True,
                         metrics={'j': current_density, 'twist': twist})

    vectors = {'B': result['b'], 'B_rtp': result['b_rtp']}
    radius = result['spherical_coords'][..., 0]
    scalars = {'radius': radius, 'current_density': np.sum(result['j'] ** 2, -1) ** 0.5, 'twist': result['twist']}
    coords = result['coords']

    args = (vtk_path, coords, vectors, scalars)
    save_thread = threading.Thread(target=save_vtk, args=args)
    save_thread.start()
    save_threads.append(save_thread)

# wait for threats to finish
[t.join() for t in save_threads]
