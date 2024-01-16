import argparse
import asyncio
import glob
import os
import threading

import numpy as np
import torch
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, cartesian_to_spherical, spherical_to_cartesian
from nf2.evaluation.unpack import load_coords
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files')
parser.add_argument('--temporal_strides', type=int, default=1, required=False, help='select the time steps (default=1; all files).')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')

parser.add_argument('--radius_range', nargs='+', type=float, default=(0.999, 1.2), required=False)
parser.add_argument('--latitude_range', nargs='+', type=float, default=(0 * np.pi, 1 * np.pi), required=False)
parser.add_argument('--longitude_range', nargs='+', type=float, default=(0 * np.pi, 2 * np.pi), required=False)
parser.add_argument('--pixels_per_solRad', type=int, default=32, required=False)

args = parser.parse_args()
nf2_path = args.nf2_path
out_path = args.out_path

os.makedirs(out_path, exist_ok=True)

radius_range = tuple(args.radius_range)
latitude_range = tuple(args.latitude_range)
longitude_range = tuple(args.longitude_range)
pixels_per_solRad = args.pixels_per_solRad

assert len(radius_range) == 2, 'radius_range must be a tuple of length 2'
assert len(latitude_range) == 2, 'latitude_range must be a tuple of length 2'
assert len(longitude_range) == 2, 'longitude_range must be a tuple of length 2'

# full disk
# radius_range = (0.999, 1.2)
# latitude_range = (0 * np.pi, 1 * np.pi)
# longitude_range = (0 * np.pi, 2 * np.pi)
# pixels_per_solRad = 64

# 2106
# radius_range = (0.999, 1.1)
# latitude_range = (1.11188089, 1.30852593)
# longitude_range = (0.39223768, 0.808977)
# pixels_per_solRad = 512

# 2173
# radius_range = (1, 1.1)
# latitude_range = (0.4 * np.pi, 0.6 * np.pi)
# longitude_range = (0.75 * np.pi, 1.0 * np.pi)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

files = sorted(glob.glob(nf2_path))[::args.temporal_strides] #if os.path.isdir(nf2_path) else [nf2_path]

save_threads = []
for i, f in enumerate(files):
    vtk_path = os.path.join(out_path, os.path.basename(f).replace('.nf2', '.vtk'))
    if os.path.exists(vtk_path) and not args.overwrite:
        print(f'Skipping existing file {f}')
        continue
    print('Processing file {} of {}'.format(i + 1, len(files)))
    state = torch.load(f, map_location=device)
    model = nn.DataParallel(state['model'])

    spherical_bounds = np.stack(
        np.meshgrid(np.linspace(radius_range[0], radius_range[1], 50),
                    np.linspace(latitude_range[0], latitude_range[1], 50),
                    np.linspace(longitude_range[0], longitude_range[1], 50), indexing='ij'), -1)

    cartesian_bounds = spherical_to_cartesian(spherical_bounds)

    x_min, x_max = cartesian_bounds[..., 0].min(), cartesian_bounds[..., 0].max()
    y_min, y_max = cartesian_bounds[..., 1].min(), cartesian_bounds[..., 1].max()
    z_min, z_max = cartesian_bounds[..., 2].min(), cartesian_bounds[..., 2].max()

    coords = np.stack(
        np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixels_per_solRad)),
                    np.linspace(y_min, y_max, int((y_max - y_min) * pixels_per_solRad)),
                    np.linspace(z_max, z_min, int((z_max - z_min) * pixels_per_solRad)), indexing='ij'), -1)
    # flipped z axis
    radius = np.sqrt(np.sum(coords ** 2, -1))

    spherical_coords = cartesian_to_spherical(coords)
    condition = (spherical_coords[..., 0] >= radius_range[0]) & (spherical_coords[..., 0] < radius_range[1]) \
                & (spherical_coords[..., 1] > latitude_range[0]) & (spherical_coords[..., 1] < latitude_range[1]) \
        # & (spherical_coords[..., 2] > longitude_range[0]) & (spherical_coords[..., 2] < longitude_range[1])
    sub_coords = coords[condition]

    spatial_norm = 1
    cube_shape = coords.shape[:-1]
    sub_b, sub_j = load_coords(model, spatial_norm, state['b_norm'], sub_coords, device, progress=True,
                        compute_currents=True)
    sub_b[..., 2] *= -1  # flip z axis
    sub_j[..., 2] *= -1 # flip z axis

    b = np.zeros(cube_shape + (3,))
    b[condition] = sub_b

    j = np.zeros(cube_shape + (3,))
    j[condition] = sub_j

    spherical_coords = cartesian_to_spherical(coords)
    b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

    args = (vtk_path, {'B': b, 'B_rtp': b_spherical}, {'radius': radius, 'current_density': np.sum(j ** 2, -1) ** 0.5})
    save_thread = threading.Thread(target=save_vtk, args=args)
    save_thread.start()
    save_threads.append(save_thread)

# wait for threats to finish
[t.join() for t in save_threads]