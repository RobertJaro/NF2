import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, spherical_to_cartesian
from nf2.evaluation.unpack import load_coords

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('vtk_path', type=str, help='path to the target VTK file')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
nf2_path = args.nf2_path
strides = args.strides

nf2_path = '/Users/robert/PycharmProjects/NF2/results/extrapolation_result.nf2'
height = 1.5
resolution = 256

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state = torch.load(nf2_path, map_location=device)
model = nn.DataParallel(state['model'])

spherical_coords = np.stack(
    np.meshgrid(np.linspace(1, 1, 1),
                np.linspace(0 * np.pi, 1 * np.pi, resolution),
                np.linspace(0 * np.pi, 2 * np.pi, resolution * 2), indexing='ij'), -1)
coords = spherical_to_cartesian(spherical_coords)

spatial_norm = 1
cube_shape = coords.shape[:-1]
b = load_coords(model, spatial_norm, state['b_norm'], coords, device, progress=True)

b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

plt.figure(figsize=(10, 5))
plt.imshow(b_spherical[0, :, :, 0], cmap='gray', origin='lower', extent=[0, 2, 0, 1])
# plot grid lines
plt.grid()
plt.xlabel('Longitude [rad]')
plt.ylabel('Latitude [rad]')
plt.tight_layout()
plt.show()