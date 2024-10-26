import argparse
import pickle
from copy import copy

import h5py
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map

from nf2.evaluation.output_metrics import squashing_factor

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--hdf5_file', type=str, help='path to the hdf5 file with the B field.')
parser.add_argument('--pkl_file', type=str, help='path to the pkl file.')
parser.add_argument('--euv_file', type=str, help='path to the EUV file.')
parser.add_argument('--sharp_file', type=str, help='path to the SHARP file.')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
args = parser.parse_args()

result_file = args.result_path
hdf5_file = args.hdf5_file
pkl_file = args.pkl_file
euv_file = args.euv_file
sharp_file = args.sharp_file

# current density map
with open(pkl_file, 'rb') as file:
    output = pickle.load(file)
current_density_map = output['maps']['current_density_map']
wcs = output['info']['wcs']
Mm_per_pixel = output['info']['Mm_per_pixel']
current_density_map = current_density_map.to_value(u.G * u.cm * u.s ** -1)
map_wcs = wcs[0]

# squashing factor
f = h5py.File(hdf5_file, 'r')
b = f["B"]
res = squashing_factor(b)
q_cube = res['q']

# EUV map
euv_map = Map(euv_file)
exposure = euv_map.exposure_time.to_value(u.s)
euv_map = euv_map.reproject_to(map_wcs)

# SHARP map
sharp_map = Map(sharp_file)

j_norm = LogNorm()
cm = copy(plt.get_cmap('sdoaia131'))
cm.set_bad('black')

fig, axs = plt.subplots(4, 1, figsize=(6, 6))
#
extent = np.array([0, current_density_map.shape[0],
                   0, current_density_map.shape[1]]) * Mm_per_pixel

ax = axs[0]
sharp_im = ax.imshow(sharp_map.data, origin='lower', cmap='gray',
                     extent=extent, vmin=-3000, vmax=3000)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(sharp_im, cax=cax, label='B$_z$ [G]')
# plot rectangle
ax.plot([100, 300, 300, 100, 100], [140, 140, 40, 40, 140], 'w--')

#
ax = axs[1]
euv_im = ax.imshow(euv_map.data / exposure, origin='lower',
                   cmap=cm, extent=extent, norm=LogNorm(vmin=5, vmax=1e3))
ax.contour(sharp_map.data, levels=[-1000, 1000], colors=['black', 'white'], extent=extent, alpha=0.9, linewidths=.7)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(euv_im, cax=cax, label='SDO/AIA 131 $\AA$ [DN / s]')

#
ax = axs[2]
cd_im = ax.imshow(current_density_map.T, origin='lower',
                  cmap='inferno', extent=extent, norm=j_norm)
ax.contour(sharp_map.data, levels=[-1000, 1000], colors=['black', 'white'], extent=extent, alpha=0.9, linewidths=.7)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(cd_im, cax=cax, label='J [G cm / s]')
#

ax = axs[3]
height = int(5 // Mm_per_pixel)
q_map = q_cube[:, :, height]
q_map = np.nan_to_num(q_map, nan=1e3)
q_im = ax.imshow(q_map.T, origin='lower', cmap='gray',
                 extent=extent, norm=LogNorm(vmin=1, vmax=1e3))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(q_im, cax=cax, label='Q')

# axs[-1].set_xlabel('X [Mm]')
# [ax.set_ylabel('Y [Mm]') for ax in np.ravel(axs)]
[ax.set_xlim(100, 300) for ax in axs[1:]]
[ax.set_ylim(40, 140) for ax in axs[1:]]

fig.tight_layout(pad=0.1)
fig.savefig(result_file, dpi=300)
plt.close()
