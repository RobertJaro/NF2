import argparse
import asyncio
import glob
import os
import threading

import numpy as np
import pfsspy
import torch
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, cartesian_to_spherical, spherical_to_cartesian, \
    vector_spherical_to_cartesian
from nf2.evaluation.unpack import load_coords
from nf2.evaluation.vtk import save_vtk
from astropy import units as u

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--Br', type=str, help='path to the source radial magnetic field file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files')

parser.add_argument('--radius_range', nargs='+', type=float, default=(0.999, 2.0), required=False)
parser.add_argument('--latitude_range', nargs='+', type=float, default=(0 * np.pi, 1 * np.pi), required=False)
parser.add_argument('--longitude_range', nargs='+', type=float, default=(0 * np.pi, 2 * np.pi), required=False)
parser.add_argument('--pixels_per_solRad', type=int, default=32, required=False)

args = parser.parse_args()
out_path = args.out_path

os.makedirs(out_path, exist_ok=True)

radius_range = tuple(args.radius_range)
latitude_range = tuple(args.latitude_range)
longitude_range = tuple(args.longitude_range)
pixels_per_solRad = args.pixels_per_solRad

assert len(radius_range) == 2, 'radius_range must be a tuple of length 2'
assert len(latitude_range) == 2, 'latitude_range must be a tuple of length 2'
assert len(longitude_range) == 2, 'longitude_range must be a tuple of length 2'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mag_r_map = Map(args.Br)
mag_r_map = mag_r_map.resample([360 * 4, 180 * 4] * u.pix)
nrho = 9
rss = 2.5
pfss_in = pfsspy.Input(mag_r_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

fig, axs = plt.subplots(10, 3, figsize=(10, 10))
b_potential = pfss_out.bg
for i in range(10):
    v_min_max = np.abs(b_potential[:, :, i]).max()
    v_min_max = 500 if v_min_max > 500 else v_min_max
    axs[i, 0].imshow(b_potential[:, :, i, 0].T, vmin=-v_min_max, vmax=v_min_max, cmap='gray', origin='lower')
    axs[i, 1].imshow(b_potential[:, :, i, 1].T, vmin=-v_min_max, vmax=v_min_max, cmap='gray', origin='lower')
    im = axs[i, 2].imshow(b_potential[:, :, i, 2].T, vmin=-v_min_max, vmax=v_min_max, cmap='gray', origin='lower')
    divider = make_axes_locatable(axs[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

[ax.axis('off') for ax in axs.ravel()]
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'potential.jpg'), dpi=300)
plt.close()


vtk_path = os.path.join(out_path, 'potential.vtk')



spherical_bounds = np.stack(
    np.meshgrid(np.linspace(radius_range[0], radius_range[1], 50),
                np.linspace(latitude_range[0], latitude_range[1], 50),
                np.linspace(longitude_range[0], longitude_range[1], 50), indexing='ij'), -1)

cartesian_bounds = spherical_to_cartesian(spherical_bounds)

x_min, x_max = cartesian_bounds[..., 0].min(), cartesian_bounds[..., 0].max()
y_min, y_max = cartesian_bounds[..., 1].min(), cartesian_bounds[..., 1].max()
z_min, z_max = cartesian_bounds[..., 2].min(), cartesian_bounds[..., 2].max()

x_min, x_max = -2, 2
y_min, y_max = -2, 2
z_min, z_max = -2, 2

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
spherical_coords[..., 1] -= np.pi / 2
sub_coords = spherical_coords[condition]


cube_shape = coords.shape[:-1]
sky_sub_coords = SkyCoord(lon=sub_coords[..., 2] * u.rad,
                      lat=(sub_coords[..., 1]) * u.rad,
                      radius=sub_coords[..., 0] * u.solRad, frame=mag_r_map.coordinate_frame)
sub_b = pfss_out.get_bvec(sky_sub_coords, out_type='cartesian')
# sub_b[..., 2] *= -1  # flip z axis

b = np.zeros(cube_shape + (3,))
b[condition] = sub_b

save_vtk(vtk_path, {'B': b}, {'radius': radius})
