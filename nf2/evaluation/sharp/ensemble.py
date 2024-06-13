import argparse
import glob
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.output import CartesianOutput
from astropy import units as u

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the directory of the NF2 files. Uses glob search pattern. (<<your path>>/**/extrapolation_result.nf2)', required=True)
parser.add_argument('--ref_path', type=str, help='path to the reference map', required=False, default=None)
parser.add_argument('--ref_euv_path', type=str, help='path to the reference map', required=False, default=None)
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
parser.add_argument('--Mm_per_pixel', type=float, help='Mm per pixel', required=False, default=0.72)
args = parser.parse_args()

runs = sorted(glob.glob(args.nf2_path, recursive=True))
ref_map = Map(args.ref_path)

ref_euv_map = Map('/glade/work/rjarolim/data/nf2/13664/euv/aia.lev1_euv_12s.2024-05-09T000007Z.304.image_lev1.fits')
euv_cmap = ref_euv_map.plot_settings['cmap']
ref_euv_map = Map(ref_euv_map.data / ref_euv_map.exposure_time.to(u.s), ref_euv_map.meta)
ref_euv_map = ref_euv_map.reproject_to(ref_map.wcs)

b_ensemble = []
j_ensemble = []
for f in tqdm(runs, desc='Loading Ensemble'):
    model = CartesianOutput(f)
    out = model.load_cube(Mm_per_pixel=args.Mm_per_pixel)
    b_ensemble.append(out['b'])
    j_ensemble.append(out['j'])

Mm_per_pixel = out['Mm_per_pixel']

b_ensemble = np.stack(b_ensemble).to_value(u.G)
j_ensemble = np.stack(j_ensemble).to_value(u.G / u.s)

mean_b_vector = np.mean(b_ensemble, axis=0, keepdims=True)
b_uncertainty = np.mean(np.linalg.norm(b_ensemble - mean_b_vector, axis=-1) ** 2, axis=0) ** 0.5

mean_j_vector = np.mean(j_ensemble, axis=0, keepdims=True)
j_uncertainty = np.mean(np.linalg.norm(j_ensemble - mean_j_vector, axis=-1) ** 2, axis=0) ** 0.5

extent_xy = [0, b_uncertainty.shape[0] * Mm_per_pixel, 0, b_uncertainty.shape[1] * Mm_per_pixel]
extent_xz = [0, b_uncertainty.shape[0] * Mm_per_pixel, 0, b_uncertainty.shape[2] * Mm_per_pixel]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

im = ax.imshow(ref_map.data, cmap='gray', origin='lower', vmin=-2000, vmax=2000, extent=extent_xy)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='[G]')
ax.set_title('Magnetic Field - B$_z$')
ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')

plt.tight_layout()
plt.savefig('/glade/work/rjarolim/nf2/sharp/13664/ensemble/mag_reference.png', dpi=300, transparent=True)
plt.close()

# ax = axs[1]
# im = ax.imshow(ref_euv_map.data, cmap=euv_cmap, origin='lower', extent=extent_xy, norm=LogNorm(vmin=1))
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='3%', pad=0.05)
# fig.colorbar(im, cax=cax, orientation='vertical', label='[DN/s]')
# ax.set_title('SDO/AIA 131 $\AA$')
# ax.set_xlabel('X [Mm]')
# ax.set_ylabel('Y [Mm]')

fig, axs = plt.subplots(2, 1, figsize=(4, 4))

ax = axs[0]
im = ax.imshow(b_uncertainty.mean(2).T, cmap='viridis', origin='lower', extent=extent_xy)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
ax.contour(ref_map.data, levels=[-1500, 1500], colors=['black', 'white'], alpha=0.6, extent=extent_xy)
fig.colorbar(im, cax=cax, orientation='vertical', label='[G]')
ax.set_title('Uncertainty')
ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')

ax = axs[1]
im = ax.imshow(b_uncertainty.mean(1).T, cmap='viridis', origin='lower', extent=extent_xz)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='[G]')
ax.set_xlabel('X [Mm]')
ax.set_ylabel('Z [Mm]')


plt.tight_layout()
plt.savefig('/glade/work/rjarolim/nf2/sharp/13664/ensemble/overview.png', dpi=300, transparent=True)
plt.close()

j_norm = LogNorm()

fig, axs = plt.subplots(2, 3, figsize=(8, 3))

[ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]
[ax.set_xlabel('X [Mm]') for ax in axs[1, :]]

axs = np.ravel(axs)
for i, (f, ax) in enumerate(zip(runs, axs[:-1])):
    current_density = np.linalg.norm(j_ensemble[0], axis=-1).sum(2) * Mm_per_pixel
    im = ax.imshow(current_density.T, cmap='inferno', origin='lower', extent=extent_xy, norm=j_norm)
    ax.set_title(f'Run {i + 1}')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='[G cm / s]')

ax = axs[-1]
current_density_std = np.std(np.linalg.norm(j_ensemble, axis=-1).sum(3) * Mm_per_pixel, axis=0)
im = ax.imshow(current_density_std.T, cmap='viridis', origin='lower', extent=extent_xy, norm=j_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='[G cm / s]')
ax.set_title('Ensemble Std')

plt.tight_layout()
plt.savefig(os.path.join(args.result_path, 'j_overview.png'), dpi=300, transparent=True)
plt.close()
