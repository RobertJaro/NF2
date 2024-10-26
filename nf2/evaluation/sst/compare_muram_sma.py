import argparse
import os

import numpy as np
from astropy import units as u, constants
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import b_nabla_bz, curl
from nf2.evaluation.output import CartesianOutput
from nf2.loader.muram import MURaMSnapshot

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/disambiguation/evaluation'
os.makedirs(out_path, exist_ok=True)

Mm_per_pixel = 0.192

sma_muram_snapshot = MURaMSnapshot('/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_A/3D', iteration=399000)
sma_muram_cube = sma_muram_snapshot.load_cube(resolution=Mm_per_pixel * u.Mm / u.pix, height=50 * u.Mm)
sma_muram_b = sma_muram_cube['B']
sma_muram_BnablaBz = b_nabla_bz(sma_muram_b) / Mm_per_pixel
sma_muram_j = (curl(sma_muram_b) / Mm_per_pixel * u.G / u.Mm) * constants.c / (4 * np.pi)
sma_muram_j = sma_muram_j.to_value(u.G / u.s)

sma_1slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_sma_1slices_v01/extrapolation_result.nf2')
sma_2slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_sma_2slices_v01/extrapolation_result.nf2')
sma_2slices_amb_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_sma_2slices_ambiguous_v01/extrapolation_result.nf2')

x_min, x_max = sma_1slice_model.coord_range[0]
y_min, y_max = sma_1slice_model.coord_range[1]

Mm_per_ds = sma_1slice_model.Mm_per_ds
ds_per_pixel = Mm_per_pixel / Mm_per_ds

coords = np.stack(
    np.meshgrid(np.linspace(x_min, x_max, np.round((x_max - x_min) / ds_per_pixel + 1).astype(int)),
                np.linspace(y_min, y_max, np.round((y_max - y_min) / ds_per_pixel + 1).astype(int)),
                np.linspace(0, 50 / Mm_per_ds, np.round(50 / Mm_per_pixel + 1).astype(int)),
                indexing='ij'), -1)
sma_1slice_out = sma_1slice_model.load_coords(coords, metrics=['j', 'b_nabla_bz'], progress=True)
sma_2slice_out = sma_2slice_model.load_coords(coords, metrics=['j', 'b_nabla_bz'], progress=True)
sma_2slices_amb_out = sma_2slices_amb_model.load_coords(coords, metrics=['j', 'b_nabla_bz'], progress=True)

##############################################################################

extent = np.array([x_min, x_max, y_min, y_max]) * Mm_per_ds

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

im = ax.imshow(sma_muram_b[:, :, 0, 2].T, origin='lower', cmap='gray',
                 vmin=-1000, vmax=1000, extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')

y_slice = 200
# draw red line at slice position
ax.axvline(y_slice * Mm_per_pixel, color='red', linestyle='--')

plt.savefig(os.path.join(out_path, 'muram_sma_bz.png'))
plt.close()

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, 0, 70]

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(sma_muram_BnablaBz[y_slice, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(sma_1slice_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(sma_2slice_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
ax.set_title('NF2 - multi height')


ax = axs[3]
ax.imshow(sma_2slices_amb_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
                 cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('Y [Mm]') for ax in axs]
axs[0].set_ylabel('Z [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_sma_b_nabla_bz.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, 0, 70]
b_norm = LogNorm(vmin=1e-1, vmax=1e3)

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(np.linalg.norm(sma_muram_b[y_slice, :, :], axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(np.linalg.norm(sma_1slice_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(np.linalg.norm(sma_2slice_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('NF2 - multi height')


ax = axs[3]
ax.imshow(np.linalg.norm(sma_2slices_amb_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_sma_b.png'), dpi=300, transparent=True)
plt.close()



##############################################################################

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, y_min * Mm_per_ds, y_max * Mm_per_ds]
j_norm = LogNorm()

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(np.linalg.norm(sma_muram_j, axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(np.linalg.norm(sma_1slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(np.linalg.norm(sma_2slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - multi height')


ax = axs[3]
ax.imshow(np.linalg.norm(sma_2slices_amb_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G cm s$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_sma_j.png'), dpi=300, transparent=True)
plt.close()
