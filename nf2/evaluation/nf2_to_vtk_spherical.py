import argparse

import numpy as np
import torch
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, cartesian_to_spherical, spherical_to_cartesian
from nf2.evaluation.metric import divergence
from nf2.evaluation.unpack import load_coords
from nf2.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('vtk_path', type=str, help='path to the target VTK file')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
nf2_path = args.nf2_path
strides = args.strides
vtk_path = args.vtk_path

nf2_path = '/Users/robert/PycharmProjects/NF2/results/nice_full_disk.nf2'
vtk_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result.vtk'
vtk_j_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result_J.vtk'
vtk_spherical_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result_spherical.vtk'
pixels_per_solRad = 128

# full disk
radius_range = (0.99, 1.2)
latitude_range = (0 * np.pi, 1 * np.pi)
longitude_range = (0 * np.pi, 2 * np.pi)

# 2106
# radius_range = (1, 1.1)
# latitude_range = (1.11188089, 1.30852593)
# longitude_range = (0.39223768, 0.808977)

# 2173
# radius_range = (1, 1.1)
# latitude_range = (0.4 * np.pi, 0.6 * np.pi)
# longitude_range = (0.75 * np.pi, 1.0 * np.pi)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state = torch.load(nf2_path, map_location=device)
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
sub_b = load_coords(model, spatial_norm, state['b_norm'], sub_coords, device, progress=True, compute_currents=False)
sub_b[..., 2] *= -1 # flip z axis
# sub_j[..., 2] *= -1 # flip z axis

b = np.zeros(cube_shape + (3,))
b[condition] = sub_b

# j = np.zeros(cube_shape + (3,))
# j[(radius > 1) & (radius <= height)] = sub_j

spherical_coords = cartesian_to_spherical(coords)
b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

save_vtk(b, vtk_path, 'B', scalar=radius, scalar_name='radius')
# save_vtk(j, vtk_j_path, 'J', scalar=radius, scalar_name='radius')
save_vtk(b_spherical, vtk_spherical_path, 'B_spherical', scalar=radius, scalar_name='radius')
# Mm_per_pix=state['Mm_per_pixel'] * strides)