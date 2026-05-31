import argparse
import os

from astropy import units as u

from nf2.evaluation.output import SphericalOutput
from nf2.evaluation.vtk import save_vtk, split_vectors_scalars


def convert(nf2_path, out_path=None, radius_range=None, latitude_range=None, longitude_range=None,
            pixels_per_solRad=64, radians=False, metrics=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), os.path.basename(os.path.dirname(nf2_path)) + '.vtk')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    radius_range = (radius_range if radius_range is not None else [0.999, 1.3]) * u.solRad
    angle_unit = u.rad if radians else u.deg
    latitude_range = (latitude_range if latitude_range is not None else [-90, 90]) * angle_unit
    longitude_range = (longitude_range if longitude_range is not None else [0, 360]) * angle_unit
    resolution = pixels_per_solRad * u.pix / u.solRad
    metrics = ['j'] if metrics is None else metrics

    output = SphericalOutput(nf2_path)
    result = output.load(radius_range, latitude_range, longitude_range, resolution,
                         progress=kwargs.pop('progress', True), metrics=metrics)

    vectors = {'B': result['b']}
    metrics_out = result.get('metrics', {})
    scalars = {'radius': result['spherical_coords'][..., 0]}
    if 'b_rtp' in result:
        scalars.update({
            'B_r': result['b_rtp'][..., 0],
            'B_theta': result['b_rtp'][..., 1],
            'B_phi': result['b_rtp'][..., 2],
        })
    metric_vectors, metric_scalars = split_vectors_scalars(metrics_out)
    vectors.update(metric_vectors)
    scalars.update(metric_scalars)
    save_vtk(out_path, result['coords'], vectors, scalars)


def main():
    parser = argparse.ArgumentParser(description='Convert spherical NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, required=True, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', default=None)
    parser.add_argument('--radius_range', nargs=2, type=float, default=[0.999, 1.3])
    parser.add_argument('--latitude_range', nargs=2, type=float, default=[-90, 90])
    parser.add_argument('--longitude_range', nargs=2, type=float, default=[0, 360])
    parser.add_argument('--radians', action='store_true', help='latitude and longitude are in radians')
    parser.add_argument('--pixels_per_solRad', type=int, default=64)
    parser.add_argument('--metrics', type=str, nargs='*', default=['j'])
    args = parser.parse_args()

    convert(**vars(args), progress=True)


if __name__ == '__main__':
    main()
