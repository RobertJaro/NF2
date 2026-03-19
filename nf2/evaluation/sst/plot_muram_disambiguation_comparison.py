import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, ListedColormap

from nf2.loader.muram import read_muram_slice

out_path = '/glade/work/rjarolim/nf2/topology/results/disambiguation'

os.makedirs(out_path, exist_ok=True)
ambiguous_data = np.load('/glade/work/rjarolim/nf2/topology/results/muram_mfr_2slices_ambiguous_v01.npz')

sl, Nvar, shape, time = read_muram_slice(
    '/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/2D/tau_slice_1.000.474000')
bz = sl[5, :, :] * np.sqrt(4 * np.pi)
bx = sl[6, :, :] * np.sqrt(4 * np.pi)
by = sl[7, :, :] * np.sqrt(4 * np.pi)
b0 = np.stack([bx, by, bz], axis=-1)

b0_disambiguated = ambiguous_data['b'][:, :, 0]

b_mag = np.linalg.norm(b0_disambiguated, axis=-1)
azi = np.arctan2(-b0_disambiguated[..., 0], b0_disambiguated[..., 1])

disambiguation_mask = (azi % 2 * np.pi) > np.pi
disambiguation_alpha = np.clip(b_mag / 1000, 0, 1)

muram_azi = np.arctan2(-b0[..., 0], b0[..., 1])
muram_mask = (muram_azi % 2 * np.pi) > np.pi
muram_alpha = np.clip(np.linalg.norm(b0, axis=-1) / 1000, 0, 1)

b_norm = Normalize(vmin=-1000, vmax=1000)
plot_kwargs = {'cmap': 'gray', 'norm': b_norm, 'origin': 'lower'}

#################################################
# plot Bx, By, Bz, disambiguation map
dBx = b0_disambiguated[..., 0] - b0[..., 0]
dBy = b0_disambiguated[..., 1] - b0[..., 1]
dBz = b0_disambiguated[..., 2] - b0[..., 2]

diff_norm = Normalize(vmin=-1000.0, vmax=1000.0)

fig = plt.figure(figsize=(9, 6), constrained_layout=True)
gs = fig.add_gridspec(nrows=4, ncols=3, height_ratios=[1, 1, 1, 0.08])

axs = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
caxs = [fig.add_subplot(gs[3, j]) for j in range(3)]

# ---- Row 1: Bx
axs[0][0].imshow(b0[..., 0].T, **plot_kwargs)
axs[0][0].set_title(r'B$_\mathrm{x}$')

axs[0][1].imshow(b0_disambiguated[..., 0].T, **plot_kwargs)
axs[0][1].set_title(r'B$_\mathrm{x}$ (disambiguated)')

im_dx = axs[0][2].imshow(dBx.T, cmap='RdBu_r', norm=diff_norm, origin='lower')
axs[0][2].set_title(r'$\Delta B_\mathrm{x}$')

# ---- Row 2: By
axs[1][0].imshow(b0[..., 1].T, **plot_kwargs)
axs[1][0].set_title(r'B$_\mathrm{y}$')

axs[1][1].imshow(b0_disambiguated[..., 1].T, **plot_kwargs)
axs[1][1].set_title(r'B$_\mathrm{y}$ (disambiguated)')

im_dy = axs[1][2].imshow(dBy.T, cmap='RdBu_r', norm=diff_norm, origin='lower')
axs[1][2].set_title(r'$\Delta B_\mathrm{y}$')

# ---- Row 3: Bz
im_bz = axs[2][0].imshow(b0[..., 2].T, **plot_kwargs)
axs[2][0].set_title(r'B$_\mathrm{z}$')

im_bz_d = axs[2][1].imshow(b0_disambiguated[..., 2].T, **plot_kwargs)
axs[2][1].set_title(r'B$_\mathrm{z}$ (disambiguated)')

im_dz = axs[2][2].imshow(dBz.T, cmap='RdBu_r', norm=diff_norm, origin='lower')
axs[2][2].set_title(r'$\Delta B_\mathrm{z}$')

# ---- Colorbars
fig.colorbar(im_bz,   cax=caxs[0], orientation='horizontal',
             label='Magnetic Field Strength [G]')
fig.colorbar(im_bz_d, cax=caxs[1], orientation='horizontal',
             label='Magnetic Field Strength [G]')
fig.colorbar(im_dz,   cax=caxs[2], orientation='horizontal',
             label=r'Signed Difference [G]')

fig.savefig(os.path.join(out_path, "muram_disambiguation_comparison.png"),
            dpi=100, transparent=True)
plt.close(fig)

#################################################
# plot azimuth flip masks
cmap = ListedColormap(['blue', 'red'])

fig = plt.figure(figsize=(6, 3), constrained_layout=True)
gs = fig.add_gridspec(
    nrows=2, ncols=2,
    height_ratios=[1, 0.10],  # bottom row reserved for colorbars
    width_ratios=[1, 1]
)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
cax0 = fig.add_subplot(gs[1, 0])
cax1 = fig.add_subplot(gs[1, 1])

im0 = ax0.imshow(
    muram_mask.T, cmap=cmap, vmin=0, vmax=1,
    alpha=muram_alpha.T, origin='lower'
)
ax0.set_title('Ground-Truth')

im1 = ax1.imshow(
    disambiguation_mask.T, cmap=cmap, vmin=0, vmax=1,
    alpha=disambiguation_alpha.T, origin='lower'
)
ax1.set_title('Disambiguation Result')

cb0 = fig.colorbar(im0, cax=cax0, orientation='horizontal', ticks=[0, 1])
cb0.set_label('Azimuth Flip')

cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal', ticks=[0, 1])
cb1.set_label('Azimuth Flip')

fig.savefig(os.path.join(out_path, 'muram_disambiguation_masks.png'),
            dpi=100, transparent=True)
plt.close(fig)
