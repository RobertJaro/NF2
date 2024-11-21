import argparse
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from nf2.evaluation.output import CartesianOutput, HeightTransformOutput

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluate SST extrapolation.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/sst/evaluation_eruption'
os.makedirs(out_path, exist_ok=True)

pre_sst = fits.getdata('/glade/work/rjarolim/data/SST/panorama_8542_StkI.fits')[10]
post_sst = fits.getdata('/glade/work/rjarolim/data/SST/panorama1050_8542_StkI.fits')[10]
post_sst = np.flip(post_sst, axis=0)
sst_Mm_per_pixel = 0.09 / 2

norm = Normalize()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(pre_sst[512:2048, 512:2048], origin='lower', cmap='gray', norm=norm)
axs[1].imshow(post_sst[512:2048, 512:2048], origin='lower', cmap='gray', norm=norm)
# plot axis grid
[ax.grid() for ax in axs]
fig.tight_layout()
fig.savefig(os.path.join(out_path, 'sst_shift_ref.png'), dpi=300, transparent=True)
plt.close()


########################################################################################################################
# find best shift between pre and post sst based on cross correlation
def _correlation_coefficient(patch1, patch2):
    product = np.nanmean((patch1 - np.nanmean(patch1)) * (patch2 - np.nanmean(patch2)))
    stds = np.nanstd(patch1) * np.nanstd(patch2)
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


x_shifts = np.arange(390, 400, 1, dtype=int)  # final shift is 396
y_shifts = np.arange(-50, -40, 1, dtype=int)  # final shift is -48
shifts = np.array(np.meshgrid(x_shifts, y_shifts)).T.reshape(-1, 2)

cc_coeffs = []
for x_shift, y_shift in tqdm(shifts.reshape(-1, 2)):
    shifted_pre = pre_sst[512:2048, 512:2048]
    shifted_post = post_sst[512 + y_shift:2048 + y_shift, 512 + x_shift:2048 + x_shift]
    cc = _correlation_coefficient(shifted_pre.flatten(), shifted_post.flatten())
    cc_coeffs.append([cc, x_shift, y_shift])

# find best cross correlation
cc_coeffs = np.array(cc_coeffs)
best_shift = cc_coeffs[np.argmax(cc_coeffs[:, 0])]

best_cc = best_shift[0]
sst_x_shift = int(best_shift[1])
sst_y_shift = int(best_shift[2])

sst_x_shift_Mm = sst_x_shift * sst_Mm_per_pixel
sst_y_shift_Mm = sst_y_shift * sst_Mm_per_pixel

print(f'Best shift ({best_cc}): x={sst_x_shift}, y={sst_y_shift}')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(pre_sst[512:2048, 512:2048], origin='lower', cmap='gray', norm=norm)
axs[1].imshow(post_sst[512 + sst_y_shift:2048 + sst_y_shift, 512 + sst_x_shift:2048 + sst_x_shift],
              origin='lower', cmap='gray', norm=norm)
# plot axis grid
[ax.grid() for ax in axs]
fig.tight_layout()
fig.savefig(os.path.join(out_path, 'sst_shift.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################

pre_nf2_file = '/glade/work/rjarolim/nf2/sst/13392_0851_2slices_v01_bu/extrapolation_result.nf2'
post_nf2_file = '/glade/work/rjarolim/nf2/sst/13392_1050_2slices_v02/extrapolation_result.nf2'

pre_model = CartesianOutput(pre_nf2_file)
post_model = CartesianOutput(post_nf2_file)

Mm_per_pixel = 0.36
height_range = [0, 40]

pre_out = pre_model.load_cube(height_range=height_range, Mm_per_pixel=Mm_per_pixel, metrics=['j', 'b_nabla_bz'],
                              progress=True)
post_out = post_model.load_cube(height_range=height_range, Mm_per_pixel=Mm_per_pixel, metrics=['j', 'b_nabla_bz'],
                                progress=True)

cm_per_pixel = Mm_per_pixel * 1e8
pre_j = np.linalg.norm(pre_out['metrics']['j'], axis=-1).to_value(u.G / u.s).sum(2) * cm_per_pixel
post_j = np.linalg.norm(post_out['metrics']['j'], axis=-1).to_value(u.G / u.s).sum(2) * cm_per_pixel

pre_bz = pre_out['b'][:, :, 0, 2].to_value(u.G)
post_bz = post_out['b'][:, :, 0, 2].to_value(u.G)

# get height surfaces
pre_height_model = HeightTransformOutput(pre_nf2_file)
post_height_model = HeightTransformOutput(post_nf2_file)

pre_height_out = pre_height_model.load_height_mapping()
post_height_out = post_height_model.load_height_mapping()

########################################################################################################################
y_slice_Mm = 53
y_slice_pre = int(y_slice_Mm / Mm_per_pixel)
y_slice_post = int((y_slice_Mm + sst_x_shift_Mm) / Mm_per_pixel)
y_range = [55, 85]

x_min, x_max = pre_model.coord_range[0] * pre_model.Mm_per_ds
y_min, y_max = pre_model.coord_range[1] * pre_model.Mm_per_ds
pre_extent = np.array([x_min, x_max, y_min, y_max])
pre_extent_yz = np.array([y_min, y_max, *height_range])

x_min, x_max = post_model.coord_range[0] * post_model.Mm_per_ds
y_min, y_max = post_model.coord_range[1] * post_model.Mm_per_ds
post_extent = np.array([x_min - sst_x_shift_Mm, x_max - sst_x_shift_Mm,
                        y_min - sst_y_shift_Mm, y_max - sst_y_shift_Mm])
post_extent_yz = np.array([y_min - sst_y_shift_Mm, y_max - sst_y_shift_Mm, *height_range])

b_norm = LogNorm(vmin=10)
j_norm = LogNorm()
sst_norm = Normalize(vmin=0, vmax=3e-8)

########################################################################################################################

fig, axs = plt.subplots(5, 2, figsize=(8, 8))

ax = axs[0, 0]
im = ax.imshow(pre_sst, origin='lower', cmap='gray', extent=pre_extent, norm=sst_norm)
ax.contour(pre_bz.T, levels=[-200, 200], colors=['blue', 'red'], extent=pre_extent, linewidths=1)

ax = axs[0, 1]
im = ax.imshow(post_sst, origin='lower', cmap='gray', extent=post_extent, norm=sst_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='8542 ${\AA}$ [DN]')
ax.contour(post_bz.T, levels=[-200, 200], colors=['blue', 'red'], extent=post_extent, linewidths=1)

ax = axs[1, 0]
im = ax.imshow(pre_j.T, origin='lower', extent=pre_extent, cmap='inferno', norm=j_norm)
ax.contour(pre_bz.T, levels=[-200, 200], colors=['black', 'white'], extent=pre_extent, linewidths=1)

ax = axs[1, 1]
im = ax.imshow(post_j.T, origin='lower', extent=post_extent, cmap='inferno', norm=j_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G cm s$^{-1}$]')
ax.contour(post_bz.T, levels=[-200, 200], colors=['black', 'white'], extent=post_extent, linewidths=1)

ax = axs[2, 0]
ax.imshow(pre_out['metrics']['b_nabla_bz'][y_slice_pre, :, :].T, origin='lower',
          cmap='RdBu_r', vmin=-.1, vmax=.1, extent=pre_extent_yz)

ax = axs[2, 1]
im = ax.imshow(post_out['metrics']['b_nabla_bz'][y_slice_post, :, :].T, origin='lower',
               cmap='RdBu_r', vmin=-.1, vmax=.1, extent=post_extent_yz)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot  \nabla \hat{B}_z$ [G Mm$^{-1}$]')

ax = axs[3, 0]
im = ax.imshow(np.linalg.norm(pre_out['b'][y_slice_pre, :, :].to_value(u.G), axis=-1).T, origin='lower',
               extent=pre_extent_yz, norm=b_norm, cmap='jet')

ax = axs[3, 1]
im = ax.imshow(np.linalg.norm(post_out['b'][y_slice_post, :, :].to_value(u.G), axis=-1).T, origin='lower',
               extent=post_extent_yz, norm=b_norm, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')

# add height surfaces
for height_mapping in pre_height_out:
    h_coords = height_mapping['coords']
    h_y_slice = np.argmin(np.abs(h_coords[:, 0, 0, 0].to_value(u.Mm) - y_slice_Mm))
    axs[2, 0].plot(h_coords[h_y_slice, :, 0, 1].to_value(u.Mm), h_coords[h_y_slice, :, 0, 2].to_value(u.Mm),
            linestyle='--', color='black')

for height_mapping in post_height_out:
    h_coords = height_mapping['coords']
    h_x = h_coords[..., 0] - sst_x_shift_Mm * u.Mm
    h_y = h_coords[..., 1] - sst_y_shift_Mm * u.Mm
    h_y_slice = np.argmin(np.abs(h_x[:, 0, 0].to_value(u.Mm) - y_slice_Mm))
    axs[2, 1].plot(h_y[h_y_slice, :, 0].to_value(u.Mm), h_coords[h_y_slice, :, 0, 2].to_value(u.Mm),
            linestyle='--', color='black')

# plot quivers for B
byz_pre = pre_out['b'][y_slice_pre, :, :, 1:].to_value(u.G)
coord = pre_out['coords'][y_slice_pre, :, :, 1:] * pre_model.Mm_per_ds
coord_q = block_reduce(coord, (3, 3, 1), np.mean)
b_q = block_reduce(byz_pre, (3, 3, 1), np.mean)
b_q = b_q / np.linalg.norm(b_q, axis=-1)[..., None]
axs[2, 0].quiver(coord_q[..., 0], coord_q[..., 1], b_q[..., 0], b_q[..., 1], color='black', scale=50)

byz_post = post_out['b'][y_slice_post, :, :, 1:].to_value(u.G)
coord = post_out['coords'][y_slice_post, :, :, 1:] * post_model.Mm_per_ds
coord[..., 0] -= sst_y_shift_Mm
coord_q = block_reduce(coord, (3, 3, 1), np.mean)
b_q = block_reduce(byz_post, (3, 3, 1), np.mean)
b_q = b_q / np.linalg.norm(b_q, axis=-1)[..., None]
axs[2, 1].quiver(coord_q[..., 0], coord_q[..., 1], b_q[..., 0], b_q[..., 1], color='black', scale=50)

# draw magenta line at slice position
[ax.plot([y_slice_Mm, y_slice_Mm], y_range, color='magenta', linestyle='--')
 for ax in axs[:2, :].flatten()]

# adjust xy limits
# [ax.set_xlim(10, 130) for ax in axs[:2, :].flatten()]
# [ax.set_ylim(20, 100) for ax in axs[:2, :].flatten()]
#
# # adjust yz limits
# [ax.set_xlim(y_range) for ax in axs[2:, :].flatten()]
# [ax.set_ylim(0, 20) for ax in axs[2:, :].flatten()]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_eruption.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################
# plot height surfaces
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
height_mapping = pre_height_out[0]
im = ax.imshow(height_mapping['coords'][..., 0, 2].to_value(u.Mm).T, origin='lower', cmap='viridis', extent=pre_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')
ax.set_title('Pre-eruption')

ax = axs[1]
height_mapping = post_height_out[0]
im = ax.imshow(height_mapping['coords'][..., 0, 2].to_value(u.Mm).T, origin='lower', cmap='viridis', extent=post_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')
ax.set_title('Post-eruption')

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_eruption_height.png'), dpi=300, transparent=True)
plt.close()
