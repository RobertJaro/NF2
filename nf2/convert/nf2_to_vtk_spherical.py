import argparse
import os

import numpy as np
from astropy import units as u

from nf2.evaluation.output import SphericalOutput
from nf2.evaluation.output_metrics import current_density, alpha
from nf2.evaluation.vtk import save_vtk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK files', required=False, default=None)
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')

    parser.add_argument('--radius_range', nargs='+', type=float, default=(0.999, 1.3), required=False)
    parser.add_argument('--latitude_range', nargs='+', type=float, default=(-90, 90), required=False)
    parser.add_argument('--longitude_range', nargs='+', type=float, default=(0, 360), required=False)
    parser.add_argument('--radians', action='store_true', help='latitude and longitude in radians', required=False,
                        default=False)
    parser.add_argument('--pixels_per_solRad', type=int, default=64, required=False)

    args = parser.parse_args()
    nf2_path = args.nf2_path
    out_path = args.out_path if args.out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), os.path.basename(os.path.dirname(nf2_path)) + '.vtk')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

    output = SphericalOutput(nf2_path)
    result = output.load(radius_range, latitude_range, longitude_range, pixels_per_solRad, progress=True,
                         metrics={'j': current_density, 'alpha': alpha})

    vectors = {'B': result['b']}
    radius = result['spherical_coords'][..., 0]
    metrics = result['metrics']
    scalars = {'radius': radius,
               'B_r': result['b_rtp'][..., 0],
               'B_theta': result['b_rtp'][..., 1],
               'B_phi': result['b_rtp'][..., 2],
               'current_density': np.sum(metrics['j'] ** 2, -1) ** 0.5,
               'alpha': metrics['alpha']}
    coords = result['coords']

    save_vtk(out_path, coords, vectors, scalars)
