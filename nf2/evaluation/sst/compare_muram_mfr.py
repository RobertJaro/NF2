import argparse
import os

import numpy as np
from astropy import units as u, constants
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import block_reduce

from nf2.evaluation.metric import b_nabla_bz, curl, b_nabla_bz_v2
from nf2.evaluation.output import CartesianOutput
from nf2.loader.muram import MURaMSnapshot

# if __name__ == '__main__':
# parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
# parser.add_argument('--output', type=str, help='output path.')
# args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/disambiguation/evaluation'
os.makedirs(out_path, exist_ok=True)


Mm_per_pixel = 0.192 #0.192

mfr_muram_snapshot = MURaMSnapshot('/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D', iteration=474000)
mfr_muram_cube = mfr_muram_snapshot.load_cube(resolution=Mm_per_pixel * u.Mm / u.pix, height=10 * u.Mm)
mfr_muram_b = mfr_muram_cube['B']
mfr_muram_BnablaBz = b_nabla_bz(mfr_muram_b) / Mm_per_pixel
mfr_muram_j = (curl(mfr_muram_b) / Mm_per_pixel * u.G / u.Mm) * constants.c / (4 * np.pi)
mfr_muram_j = mfr_muram_j.to_value(u.G / u.s)

mfr_1slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_mfr_1slices_v01/extrapolation_result.nf2')
mfr_2slice_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_mfr_2slices_v01/extrapolation_result.nf2')
mfr_2slices_amb_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/disambiguation/muram_mfr_2slices_ambiguous_v01/extrapolation_result.nf2')

x_min, x_max = mfr_1slice_model.coord_range[0]
y_min, y_max = mfr_1slice_model.coord_range[1]

Mm_per_ds = mfr_1slice_model.Mm_per_ds
ds_per_pixel = Mm_per_pixel / Mm_per_ds

mfr_1slice_out = mfr_1slice_model.load_cube([0, 10], metrics=['j', 'b_nabla_bz'], progress=True, Mm_per_pixel=Mm_per_pixel)
mfr_2slice_out = mfr_2slice_model.load_cube([0, 10], metrics=['j', 'b_nabla_bz'], progress=True, Mm_per_pixel=Mm_per_pixel)
mfr_2slices_amb_out = mfr_2slices_amb_model.load_cube([0, 10], metrics=['j', 'b_nabla_bz'], progress=True, Mm_per_pixel=Mm_per_pixel)

##############################################################################
x_slice = 175
extent = np.array([x_min, x_max, y_min, y_max]) * Mm_per_ds

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

im = ax.imshow(mfr_muram_b[:, :, 0, 2].T, origin='lower', cmap='gray',
                 vmin=-1000, vmax=1000, extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')


# draw red line at slice position
ax.axvline(x_slice * Mm_per_pixel, color='red', linestyle='--')

plt.savefig(os.path.join(out_path, 'muram_mfr_bz.png'))
plt.close()

extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 10]


def _plot(ax, b, coord, bnbz, title):
    coord = coord[x_slice]
    #
    im = ax.imshow(bnbz[x_slice, :, :].T, origin='lower', cmap='RdBu_r', vmin=-.1, vmax=.1, extent=extent)
    ax.set_title(title)
    #
    byz_pre = b[x_slice, :, :, 1:]
    coord_q = coord[::2, ::2]  # block_reduce(coord, (3, 3, 1), np.mean)
    b_q = byz_pre[::2, ::2]  # block_reduce(byz_pre, (3, 3, 1), np.mean)
    b_q = b_q / np.linalg.norm(b_q, axis=-1)[..., None]
    print(b_q.shape, coord_q.shape)
    ax.quiver(coord_q[..., 0], coord_q[..., 1], b_q[..., 0], b_q[..., 1], color='darkgray', scale=40, pivot='middle')
    #
    return im

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

coord = np.stack(np.meshgrid(np.linspace(y_min * Mm_per_ds, y_max * Mm_per_ds, mfr_muram_b.shape[1]),
                    np.linspace(0, 10, mfr_muram_b.shape[2]), indexing='ij'), -1)
_plot(axs[0], mfr_muram_b, coord, mfr_muram_BnablaBz, 'MURaM')

coord = mfr_1slice_out['coords'] * Mm_per_ds
_plot(axs[1], mfr_1slice_out['b'].to_value(u.G), coord, mfr_1slice_out['metrics']['b_nabla_bz'], 'NF2 - single height')

_plot(axs[2], mfr_2slice_out['b'].to_value(u.G), coord, mfr_2slice_out['metrics']['b_nabla_bz'], 'NF2 - multi height')

im = _plot(axs[3], mfr_2slices_amb_out['b'].to_value(u.G), coord, mfr_2slices_amb_out['metrics']['b_nabla_bz'], 'NF2 - multi height (ambiguous)')

divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')


[ax.set_xlabel('Y [Mm]') for ax in axs]
axs[0].set_ylabel('Z [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

[ax.set_xlim([25, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_mfr_b_nabla_bz.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, 0, 10]
b_norm = LogNorm(vmin=1e-1, vmax=1e3)

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(np.linalg.norm(mfr_muram_b[x_slice, :, :], axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(np.linalg.norm(mfr_1slice_out['b'][x_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(np.linalg.norm(mfr_2slice_out['b'][x_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
ax.set_title('NF2 - multi height')


ax = axs[3]
ax.imshow(np.linalg.norm(mfr_2slices_amb_out['b'][x_slice, :, :].to_value(u.G), axis=-1).T, origin='lower', extent=extent, norm=b_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_mfr_b.png'), dpi=300, transparent=True)
plt.close()



##############################################################################

extent = [x_min * Mm_per_ds, x_max * Mm_per_ds, y_min * Mm_per_ds, y_max * Mm_per_ds]
j_norm = LogNorm()

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(np.linalg.norm(mfr_muram_j, axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(np.linalg.norm(mfr_1slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(np.linalg.norm(mfr_2slice_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - multi height')


ax = axs[3]
ax.imshow(np.linalg.norm(mfr_2slices_amb_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G cm s$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'muram_mfr_j.png'), dpi=300, transparent=True)
plt.close()
