import argparse

import numpy as np
import torch
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, cartesian_to_spherical, spherical_to_cartesian
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

nf2_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result.nf2'
vtk_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result.vtk'
vtk_spherical_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result_spherical.vtk'
strides = 1
height = 1.2
pixels_per_solRad = 64

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state = torch.load(nf2_path, map_location=device)
model = nn.DataParallel(state['model'])

# spherical_bounds = np.stack(
#     np.meshgrid(np.linspace((1 - 0.2) * np.pi, (1 + 0.05) * np.pi, 50),
#                 np.linspace((0.5 - 0.1) * np.pi, (0.5 + 0.1) * np.pi, 50),
#                 np.linspace(1, height, 50), indexing='ij'), -1)
spherical_bounds = np.stack(
    np.meshgrid(np.linspace(2, 5, 50),
                np.linspace((0.5 - 0.3) * np.pi, (0.5 + 0.3) * np.pi, 50),
                np.linspace(1, height, 50), indexing='ij'), -1)
cartesian_bounds = spherical_to_cartesian(spherical_bounds)

x_min, x_max = cartesian_bounds[..., 0].min(), cartesian_bounds[..., 0].max()
y_min, y_max = cartesian_bounds[..., 1].min(), cartesian_bounds[..., 1].max()
z_min, z_max = cartesian_bounds[..., 2].min(), cartesian_bounds[..., 2].max()

coords = np.stack(
    np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixels_per_solRad)),
                np.linspace(y_min, y_max, int((y_max - y_min) * pixels_per_solRad)),
                np.linspace(z_min, z_max, int((z_max - z_min) * pixels_per_solRad)), indexing='ij'), -1)

spatial_norm = 1
cube_shape = coords.shape[:-1]
b = load_coords(model, cube_shape, spatial_norm, state['b_norm'], coords, device, progress=True)
radius = np.sqrt(np.sum(coords ** 2, -1))
b[(radius < 1) | (radius > height)] = 0

radius = np.sqrt(np.sum(coords ** 2, -1))
b_spherical = vector_cartesian_to_spherical(b, coords)

save_vtk(b, vtk_path, 'B', scalar=radius, scalar_name='radius')
save_vtk(b_spherical, vtk_spherical_path, 'B_spherical', scalar=radius, scalar_name='radius')
# Mm_per_pix=state['Mm_per_pixel'] * strides)
