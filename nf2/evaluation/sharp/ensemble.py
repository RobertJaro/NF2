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
parser.add_argument('--ref_path', type=str, help='path to the reference map')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
parser.add_argument('--Mm_per_pixel', type=float, help='Mm per pixel', required=False, default=0.72)
args = parser.parse_args()

result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

runs = sorted(glob.glob(args.nf2_path, recursive=True))
ref_map = Map(args.ref_path)

b_ensemble = []
j_ensemble = []
for f in tqdm(runs, desc='Loading Ensemble'):
    model = CartesianOutput(f)
    out = model.load_cube(Mm_per_pixel=args.Mm_per_pixel, metrics=['j'])
    b_ensemble.append(out['b'])
    j_ensemble.append(out['metrics']['j'])

Mm_per_pixel = out['Mm_per_pixel']

b_ensemble = np.stack(b_ensemble).to_value(u.G)
j_ensemble = np.stack(j_ensemble).to_value(u.G / u.s)

mean_b_vector = np.mean(b_ensemble, axis=0, keepdims=True)
b_uncertainty = np.mean(np.linalg.norm(b_ensemble - mean_b_vector, axis=-1) ** 2, axis=0) ** 0.5

mean_j_vector = np.mean(j_ensemble, axis=0, keepdims=True)
j_uncertainty = np.mean(np.linalg.norm(j_ensemble - mean_j_vector, axis=-1) ** 2, axis=0) ** 0.5

extent_xy = [0, b_uncertainty.shape[0] * Mm_per_pixel, 0, b_uncertainty.shape[1] * Mm_per_pixel]
extent_xz = [0, b_uncertainty.shape[0] * Mm_per_pixel, 0, b_uncertainty.shape[2] * Mm_per_pixel]

fig, axs = plt.subplots(1, 3, figsize=(12, 2))

ax = axs[0]
im = ax.imshow(ref_map.data, cmap='gray', origin='lower', vmin=-2000, vmax=2000, extent=extent_xy)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='B$_z$ [G]')

ax = axs[1]
im = ax.imshow(b_uncertainty.mean(2).T, cmap='viridis', origin='lower', extent=extent_xy, vmin=0, vmax=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='$\delta$B [G]')

ax = axs[2]
b_norm = np.linalg.norm(mean_b_vector[0], axis=-1)
relative_uncertainty = b_uncertainty / (b_norm + 1e-6) * 100
im = ax.imshow(np.nanmean(relative_uncertainty, 2).T,
               cmap='cividis', origin='lower', extent=extent_xy, vmin=0, vmax=30)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\delta$B$_{\text{relative}}$ [%]')

plt.tight_layout()
plt.savefig(os.path.join(result_path, 'uncertainty.png'), dpi=300, transparent=True)
plt.close()

j_norm = LogNorm()

fig, axs = plt.subplots(1, 6, figsize=(21, 3))

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

plt.savefig(os.path.join(result_path, 'j_overview.png'), dpi=300, transparent=True)
plt.close()
