import argparse
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.output import CartesianOutput

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluate SST extrapolation.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/sst/evaluation_eruption'
os.makedirs(out_path, exist_ok=True)

pre_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/sst/13392_2slices_0851_v01/extrapolation_result.nf2')
post_model = CartesianOutput(
    '/glade/work/rjarolim/nf2/sst/13392_2slices_1050_v03/extrapolation_result.nf2')

Mm_per_pixel = 0.36
pre_out = pre_model.load_cube(height_range=[0, 40], Mm_per_pixel=Mm_per_pixel, metrics=['j', 'b_nabla_bz'], progress=True)
post_out = post_model.load_cube(height_range=[0, 40], Mm_per_pixel=Mm_per_pixel, metrics=['j', 'b_nabla_bz'], progress=True)

########################################################################################################################
y_slice_pre = 150
y_slice_post = 200
y_range = [50, 90]


fig, axs = plt.subplots(2, 1, figsize=(10, 5))

ax = axs[0]
im = ax.imshow(pre_out['b'][:, :, 0, 2].to_value(u.G).T, origin='lower', cmap='gray',
               vmin=-2000, vmax=2000)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')
ax.axvline(y_slice_pre, color='red', linestyle='--')

ax = axs[1]
im = ax.imshow(post_out['b'][:, :, 0, 2].to_value(u.G).T, origin='lower', cmap='gray')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Intensity')
ax.axvline(y_slice_post, color='red', linestyle='--')

# draw red line at slice position
# [ax.plot([y_slice, y_slice], [y_range[0], y_range[1]], color='red', linestyle='--') for ax
#  in axs]


plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_bz.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(pre_out['metrics']['b_nabla_bz'][y_slice_pre, :, :].T, origin='lower',
          cmap='RdBu_r', vmin=-.1, vmax=.1)
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(post_out['metrics']['b_nabla_bz'][y_slice_post, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1)
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot  \nabla \hat{B}_z$ [G Mm$^{-1}$]')

[ax.set_xlabel('Y [Mm]') for ax in axs]
axs[0].set_ylabel('Z [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
# [ax.set_xlim(y_range) for ax in axs]
# [ax.set_ylim([0, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_b_nabla_bz.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

b_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(pre_out['b'][y_slice_pre, :, :].to_value(u.G), axis=-1).T, origin='lower',
          norm=b_norm, cmap='jet')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(np.linalg.norm(post_out['b'][y_slice_post, :, :].to_value(u.G), axis=-1).T, origin='lower',
               norm=b_norm, cmap='jet')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
# [ax.set_xlim(y_range) for ax in axs]
# [ax.set_ylim([0, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_b.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

j_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(pre_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8,
          origin='lower', norm=j_norm, cmap='inferno')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(
    np.linalg.norm(post_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum(2).T * Mm_per_pixel * 1e8,
    origin='lower', norm=j_norm, cmap='inferno')
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

b_norm = LogNorm()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
ax.imshow(np.linalg.norm(pre_out['metrics']['j'][y_slice_pre, :, :].to_value(u.G / u.s), axis=-1).T, origin='lower',
          norm=b_norm, cmap='inferno')
ax.set_title('NF2 - single height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_axis_off()

ax = axs[1]
im = ax.imshow(np.linalg.norm(post_out['metrics']['j'][y_slice_post, :, :].to_value(u.G / u.s), axis=-1).T,
               origin='lower', norm=b_norm, cmap='inferno')
ax.set_title('NF2 - multi height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G s$^{-1}$]')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
# [ax.set_xlim(y_range) for ax in axs]
# [ax.set_ylim([0, 40]) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_j_slice.png'), dpi=300, transparent=True)
plt.close()

##############################################################################

extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, 0, 70]
b_norm = LogNorm(vmin=10)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

im = ax.imshow(np.linalg.norm(post_out['b'][y_slice, :, :].to_value(u.G), axis=-1).T, origin='lower',
               extent=extent, norm=b_norm, cmap='jet')
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
