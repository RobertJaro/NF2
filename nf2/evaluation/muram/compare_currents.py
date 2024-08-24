import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import curl
from nf2.evaluation.output import CartesianOutput
from nf2.loader.muram import MURaMSnapshot

from astropy import constants as const
from astropy import units as u

muram_path = '/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D'
iteration = 474000


muram_snapshot = MURaMSnapshot(muram_path, iteration)
muram_cube = muram_snapshot.load_cube()
b_muram = muram_cube['B']

nf2_single_height = CartesianOutput('/glade/work/rjarolim/nf2/multi_height/muram_mfr_single_height_v02/extrapolation_result.nf2')
nf2_multi_height = CartesianOutput('/glade/work/rjarolim/nf2/multi_height/muram_mfr_v03/extrapolation_result.nf2')
nf2_disambiguation = CartesianOutput('/glade/work/rjarolim/nf2/multi_height/muram_mfr_disambiguation_v04/extrapolation_result.nf2')

nf2_res = 0.192
muram_res = 0.192
res_single_height = nf2_single_height.load_cube(Mm_per_pixel=nf2_res, progress=True, height_range=[0, b_muram.shape[2] * muram_res])
res_multi_height = nf2_multi_height.load_cube(Mm_per_pixel=nf2_res, progress=True, height_range=[0, b_muram.shape[2] * muram_res])
res_disambiguation = nf2_disambiguation.load_cube(Mm_per_pixel=nf2_res, progress=True, height_range=[0, b_muram.shape[2] * muram_res])

# integrated currents
j_muram = curl(b_muram) * u.G / (muram_res * u.Mm) * const.c / (4 * np.pi) # Mm_per_pixel
j_muram = (j_muram).to(u.G / u.s)

offset_z = int(5 / nf2_res)
integrated_j_single_height = np.linalg.norm(res_single_height['J'], axis=-1)[:, :, offset_z:].sum(axis=2) * nf2_single_height.Mm_per_pixel
integrated_j_multi_height = np.linalg.norm(res_multi_height['J'], axis=-1)[:, :, offset_z:].sum(axis=2) * nf2_multi_height.Mm_per_pixel
integrated_j_disambiguation = np.linalg.norm(res_disambiguation['J'], axis=-1)[:, :, offset_z:].sum(axis=2) * nf2_disambiguation.Mm_per_pixel
offset_z = int(5 / muram_res)
integrated_j_muram = np.linalg.norm(j_muram, axis=-1)[:, :, offset_z:].sum(2) * muram_res * u.Mm / u.pix

# plot integrated currents
norm = LogNorm(vmin=1e1, vmax=1e6)

fig, axs = plt.subplots(3, 4, figsize=(15, 10))


ax = axs[0, 0]
extent = [0, muram_res * integrated_j_muram.shape[0], 0, muram_res * integrated_j_muram.shape[1]]
im = ax.imshow(integrated_j_muram.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('MURaM')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[0, 1]
extent = [0, nf2_res * integrated_j_single_height.shape[0], 0, nf2_res * integrated_j_single_height.shape[1]]
im = ax.imshow(integrated_j_single_height.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Single Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[0, 2]
extent = [0, nf2_res * integrated_j_multi_height.shape[0], 0, nf2_res * integrated_j_multi_height.shape[1]]
im = ax.imshow(integrated_j_multi_height.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Multi Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[0, 3]
extent = [0, nf2_res * integrated_j_disambiguation.shape[0], 0, nf2_res * integrated_j_disambiguation.shape[1]]
im = ax.imshow(integrated_j_disambiguation.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Auto-Disambiguation')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

x_slice_Mm = 35
[a.set_xlabel('X [Mm]') for a in axs[0]]
[a.set_ylabel('Y [Mm]') for a in axs[0]]
[a.axvline(x_slice_Mm, color='white', linestyle='--') for a in axs[0]]

# plot slices
x = int(x_slice_Mm / muram_res)
muram_slice = np.linalg.norm(j_muram, axis=-1)[x]
muram_b_slice = np.linalg.norm(b_muram[..., :2], axis=-1)[x]
x = int(x_slice_Mm / nf2_res)
single_height_slice = np.linalg.norm(res_single_height['J'], axis=-1)[x]
single_height_b_slice = np.linalg.norm(res_single_height['B'][..., :2], axis=-1)[x]
multi_height_slice = np.linalg.norm(res_multi_height['J'], axis=-1)[x]
multi_height_b_slice = np.linalg.norm(res_multi_height['B'][..., :2], axis=-1)[x]
disambiguation_slice = np.linalg.norm(res_disambiguation['J'], axis=-1)[x]
disambiguation_b_slice = np.linalg.norm(res_disambiguation['B'][..., :2], axis=-1)[x]

norm = LogNorm(vmin=1e1, vmax=2e4)

ax = axs[1, 0]
extent = [0, muram_res * muram_slice.shape[0], 0, muram_res * muram_slice.shape[1]]
im = ax.imshow(muram_slice.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('MURaM')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[1, 1]
extent = [0, nf2_res * single_height_slice.shape[0], 0, nf2_res * single_height_slice.shape[1]]
im = ax.imshow(single_height_slice.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Single Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[1, 2]
extent = [0, nf2_res * multi_height_slice.shape[0], 0, nf2_res * multi_height_slice.shape[1]]
im = ax.imshow(multi_height_slice.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Multi Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[1, 3]
extent = [0, nf2_res * disambiguation_slice.shape[0], 0, nf2_res * disambiguation_slice.shape[1]]
im = ax.imshow(disambiguation_slice.value.T, cmap='plasma', norm=norm, extent=extent, origin='lower')
ax.set_title('Auto-Disambiguation')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

[a.set_xlabel('X [Mm]') for a in axs[1]]
[a.set_ylabel('Z [Mm]') for a in axs[1]]

b_norm = LogNorm(vmin=10, vmax=1e3)
ax = axs[2, 0]
scaling = (muram_b_slice.shape[1] * muram_res) ** 2
extent = [0, muram_res * muram_b_slice.shape[0], 0, muram_res * muram_b_slice.shape[1]]
im = ax.imshow(muram_b_slice.T, cmap='cividis', extent=extent, origin='lower', norm=b_norm)
ax.set_title('MURaM')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[2, 1]
scaling = (single_height_b_slice.shape[1] * nf2_res) ** 2
extent = [0, nf2_res * single_height_b_slice.shape[0], 0, nf2_res * single_height_b_slice.shape[1]]
im = ax.imshow(single_height_b_slice.value.T, cmap='cividis', extent=extent, origin='lower', norm=b_norm)
ax.set_title('Single Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[2, 2]
extent = [0, nf2_res * multi_height_b_slice.shape[0], 0, nf2_res * multi_height_b_slice.shape[1]]
im = ax.imshow(multi_height_b_slice.value.T, cmap='cividis', extent=extent, origin='lower', norm=b_norm)
ax.set_title('Multi Height')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[2, 3]
extent = [0, nf2_res * disambiguation_b_slice.shape[0], 0, nf2_res * disambiguation_b_slice.shape[1]]
im = ax.imshow(disambiguation_b_slice.value.T, cmap='cividis', extent=extent, origin='lower', norm=b_norm)
ax.set_title('Auto-Disambiguation')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

[a.set_xlabel('X [Mm]') for a in axs[2]]
[a.set_ylabel('Z [Mm]') for a in axs[2]]

fig.tight_layout()
fig.savefig('/glade/work/rjarolim/nf2/multi_height/slices.jpg', dpi=300)
plt.close(fig)

height_muram = np.linspace(0, muram_res * muram_slice.shape[1], muram_slice.shape[1])
height_nf2 = np.linspace(0, nf2_res * disambiguation_slice.shape[1], disambiguation_slice.shape[1])

profile_muram = muram_slice.mean(axis=0)
profile_single_height = single_height_slice.mean(axis=0)
profile_multi_height = multi_height_slice.mean(axis=0)
profile_disambiguation = disambiguation_slice.mean(axis=0)

profile_bh_muram = muram_b_slice.mean(axis=0)
profile_bh_single_height = single_height_b_slice.mean(axis=0)
profile_bh_multi_height = multi_height_b_slice.mean(axis=0)
profile_bh_disambiguation = disambiguation_b_slice.mean(axis=0)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

ax = axs[0]
ax.plot(profile_muram, height_muram, label='MURaM')
ax.plot(profile_single_height, height_nf2, label='Single Height')
ax.plot(profile_multi_height, height_nf2, label='Multi Height')
ax.plot(profile_disambiguation, height_nf2, label='Auto-Disambiguation')
ax.set_xlabel('Current [G/s]')
ax.set_ylabel('Z [Mm]')
ax.semilogx()
ax.set_ylim(0)

ax = axs[1]
ax.plot(profile_bh_muram, height_muram, label='MURaM')
ax.plot(profile_bh_single_height, height_nf2, label='Single Height')
ax.plot(profile_bh_multi_height, height_nf2, label='Multi Height')
ax.plot(profile_bh_disambiguation, height_nf2, label='Auto-Disambiguation')
ax.set_xlabel('B$_{||}$ [G]')
ax.set_ylabel('Z [Mm]')
ax.semilogx()
ax.set_ylim(0)

# add single legend for both plots
axs[1].legend(bbox_to_anchor=(1.1, 1.05))

fig.tight_layout()
fig.savefig('/glade/work/rjarolim/nf2/multi_height/profiles.jpg', dpi=300)
plt.close(fig)