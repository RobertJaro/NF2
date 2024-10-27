import argparse
import os

import numpy as np
from astropy import units as u, constants
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import b_nabla_bz, curl
from nf2.evaluation.output import CartesianOutput, HeightTransformOutput
from nf2.loader.muram import MURaMSnapshot

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluate SST extrapolation.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/sst/evaluation'
os.makedirs(out_path, exist_ok=True)

# select intensity data in wings
sst_data = fits.getdata('/glade/work/rjarolim/data/SST/panorama_8542_StkI.fits')[10]


mfr_1slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/sst/13392_1slices_0851_v01/extrapolation_result.nf2')
mfr_2slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/sst/13392_2slices_0851_v02/extrapolation_result.nf2')

x_min, x_max = mfr_1slice_model.coord_range[0]
y_min, y_max = mfr_1slice_model.coord_range[1]

Mm_per_pixel = 0.36
Mm_per_ds = mfr_1slice_model.Mm_per_ds
ds_per_pixel = Mm_per_pixel / Mm_per_ds

coords = np.stack(
    np.meshgrid(np.linspace(x_min, x_max, np.round((x_max - x_min) / ds_per_pixel + 1).astype(int)),
                np.linspace(y_min, y_max, np.round((y_max - y_min) / ds_per_pixel + 1).astype(int)),
                np.linspace(0, 50 / Mm_per_ds, np.round(50 / Mm_per_pixel + 1).astype(int)),
                indexing='ij'), -1)
mfr_1slice_out = mfr_1slice_model.load_coords(coords, metrics=['j', 'b_nabla_bz'], progress=True)
mfr_2slice_out = mfr_2slice_model.load_coords(coords, metrics=['j', 'b_nabla_bz'], progress=True)

height_model = HeightTransformOutput('/glade/work/rjarolim/nf2/sst/13392_2slices_0851_v02/extrapolation_result.nf2')
height_out = height_model.load_height_mapping()

########################################################################################################################
# y_slice = 470
# y_range = [90, 130]

y_slice = 150
y_range = [50, 90]

extent = np.array([x_min, x_max, y_min, y_max]) * Mm_per_ds

fig, axs = plt.subplots(2, 1, figsize=(10, 5))

ax = axs[0]
im = ax.imshow(mfr_1slice_out['b'][:, :, 0, 2].to_value(u.G).T, origin='lower', cmap='gray',
                 vmin=-2000, vmax=2000, extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')

ax = axs[1]
im = ax.imshow(sst_data, origin='lower', cmap='gray', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Intensity')

# draw red line at slice position
[ax.plot([y_slice * Mm_per_pixel, y_slice * Mm_per_pixel], [y_range[0], y_range[1]], color='red', linestyle='--') for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_bz.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################


extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 70]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))


ax = axs[0]
ax.imshow(mfr_1slice_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(mfr_2slice_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot  \nabla \hat{B}_z$ [G Mm$^{-1}$]')

[ax.set_xlabel('Y [Mm]') for ax in axs]
axs[0].set_ylabel('Z [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
[ax.set_xlim(y_range) for ax in axs]
[ax.set_ylim([0, 40]) for ax in axs]


h_coords = height_out[0]['coords']
h_y_slice = np.argmin(np.abs(h_coords[:, 0, 0, 0].to_value(u.Mm) - y_slice * Mm_per_pixel))
ax.plot(np.linspace(y_min, y_max, h_coords.shape[1]) * Mm_per_ds, h_coords[h_y_slice, :, 0, 2].to_value(u.Mm), color='black', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_b_nabla_bz.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 70]
b_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(mfr_1slice_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm, cmap='jet')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(np.linalg.norm(mfr_2slice_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm, cmap='jet')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')


[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
[ax.set_xlim(y_range) for ax in axs]
[ax.set_ylim([0, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_b.png'), dpi=300, transparent=True)
plt.close()



##############################################################################

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, y_min * Mm_per_ds, y_max * Mm_per_ds]
j_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(mfr_1slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm, cmap='inferno')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(np.linalg.norm(mfr_2slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm, cmap='inferno')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G cm s$^{-1}$]')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_j.png'), dpi=300, transparent=True)
plt.close()


##############################################################################

extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 70]
b_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(mfr_1slice_out['metrics']['j'][y_slice, :, :].to_value(u.G / u.s), axis=-1).T, origin='lower', extent=extent, norm=b_norm, cmap='inferno')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(np.linalg.norm(mfr_2slice_out['metrics']['j'][y_slice, :, :].to_value(u.G / u.s), axis=-1).T, origin='lower', extent=extent, norm=b_norm, cmap='inferno')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G s$^{-1}$]')


[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
[ax.set_xlim(y_range) for ax in axs]
[ax.set_ylim([0, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_j_slice.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 70]
b_norm = LogNorm(vmin=10)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

im = ax.imshow(np.linalg.norm(mfr_2slice_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm, cmap='jet')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')


ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')
ax.set_xlim([65 - 15, 65 + 15])
ax.set_ylim([0, 20])

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_b_multi.png'), dpi=300, transparent=True)
plt.close()