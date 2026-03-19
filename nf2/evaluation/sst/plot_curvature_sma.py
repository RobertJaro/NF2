import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import LogNorm

from nf2.evaluation.output_metrics import squashing_factor

out_path = '/glade/work/rjarolim/nf2/topology/results/curvature'
os.makedirs(out_path, exist_ok=True)

data_muram = dict(np.load('/glade/work/rjarolim/nf2/topology/results/muram_sma.npz'))
data_1slices = np.load('/glade/work/rjarolim/nf2/topology_stash4/results/muram_sma_1slices_v01.npz')
data_2slices = np.load('/glade/work/rjarolim/nf2/topology_stash4/results/muram_sma_2slices_v01.npz')
data_2slices_ambiguous = np.load('/glade/work/rjarolim/nf2/topology_stash4/results/muram_sma_2slices_ambiguous_v01.npz')

data_2slices_heights = np.load('/glade/work/rjarolim/nf2/topology/results/muram_sma_2slices_ambiguous_height_v01.pkl',
                               allow_pickle=True)
data_2slices_ambiguous_heights = np.load(
    '/glade/work/rjarolim/nf2/topology/results/muram_sma_2slices_ambiguous_height_v01.pkl', allow_pickle=True)

coords = data_1slices['coords']  # (x, y, z, 3)

x_min, x_max = coords[:, :, :, 0].min(), coords[:, :, :, 0].max()
y_min, y_max = coords[:, :, :, 1].min(), coords[:, :, :, 1].max()
z_min, z_max = coords[:, :, :, 2].min(), coords[:, :, :, 2].max()

x_slice = -13  # 37 + x_min  # Mm
# get pixel index close to x_slice in Mm
x_slice_pix = np.argmin(np.abs(np.linspace(x_min, x_max, coords.shape[0]) - x_slice))

#############################################################
# compute squashing factor
offset = int(1.5 / data_muram['Mm_per_pixel'])  # Mm -> pixels
data_muram_squashing_factor = squashing_factor(data_muram['b'][:, :, offset:], x_range=[x_slice_pix, x_slice_pix + 1])
data_1slices_squashing_factor = squashing_factor(data_1slices['b'][:, :, offset:],
                                                 x_range=[x_slice_pix, x_slice_pix + 1])
data_2slices_squashing_factor = squashing_factor(data_2slices['b'][:, :, offset:],
                                                 x_range=[x_slice_pix, x_slice_pix + 1])
data_2slices_ambiguous_squashing_factor = squashing_factor(data_2slices_ambiguous['b'][:, :, offset:],
                                                           x_range=[x_slice_pix, x_slice_pix + 1])

#############################################################
# compute MURaM heights from tau values
muram_heights = []
muram_tau = data_muram['tau']
for tau in [1e-4, 1e-6]:
    height_muram = np.argmin(np.abs(muram_tau - tau), axis=2) * data_muram['Mm_per_pixel'] * u.Mm
    muram_heights.append(height_muram)

#############################################################
# load NF2 heights
data_2slices_heights = [d['coords'][:, :, 0, 2] for d in data_2slices_heights]
data_2slices_ambiguous_heights = [d['coords'][:, :, 0, 2] for d in data_2slices_ambiguous_heights]


#############################################################
# create plot with 4 columns and 3 rows visualizing the following metrics:
# - b_nabla_bz
# - squashing factor Q
# - twist number
# add in heights of the 2-slices and ambiguous 2-slices as dashed black lines

def _plot_b_nabla_bz(data, ax, heights=None):
    im = ax.imshow(data[x_slice_pix, :, :].T,
                   origin='lower', cmap='coolwarm', vmin=-.1, vmax=.1, extent=[y_min, y_max, z_min, z_max])
    if heights is not None:
        for h in heights:
            ax.plot(np.linspace(y_min, y_max, h.shape[1]), h[x_slice_pix, :].to_value(u.Mm),
                    color='black', linestyle='--')
    return im


def _plot_squashing_factor_Q(data, ax):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='viridis',
                   extent=[y_min, y_max, z_min + offset * data_muram['Mm_per_pixel'], z_max],
                   norm=LogNorm(vmin=1, vmax=1e3))
    return im


def _plot_twist(data, ax):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='seismic',
                   extent=[y_min, y_max, z_min + offset * data_muram['Mm_per_pixel'], z_max], vmin=-1, vmax=1)
    return im


fig = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(
    3, 5,
    width_ratios=[1, 1, 1, 1, 0.05],
    wspace=0.15, hspace=0.15
)

axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(3)])

# convenience: data panels are the first 4 columns; last col is colorbar axes
panel_axs = axs[:, :4]
cax0 = axs[0, 4]
cax1 = axs[1, 4]

# ---- curvature (row 0) ----
im0 = _plot_b_nabla_bz(data_muram['b_nabla_bz'], panel_axs[0, 0], heights=muram_heights)
im0 = _plot_b_nabla_bz(data_1slices['b_nabla_bz'], panel_axs[0, 1])
im0 = _plot_b_nabla_bz(data_2slices['b_nabla_bz'], panel_axs[0, 2], heights=data_2slices_heights)
im0 = _plot_b_nabla_bz(data_2slices_ambiguous['b_nabla_bz'], panel_axs[0, 3], heights=data_2slices_ambiguous_heights)

fig.colorbar(im0, cax=cax0, location='right', label=r'$b \cdot \nabla b_z$')

# ---- squashing factor Q (row 1) ----
im1 = _plot_squashing_factor_Q(data_muram_squashing_factor['q'], panel_axs[1, 0])
im1 = _plot_squashing_factor_Q(data_1slices_squashing_factor['q'], panel_axs[1, 1])
im1 = _plot_squashing_factor_Q(data_2slices_squashing_factor['q'], panel_axs[1, 2])
im1 = _plot_squashing_factor_Q(data_2slices_ambiguous_squashing_factor['q'], panel_axs[1, 3])

fig.colorbar(im1, cax=cax1, location='right', label=r'$\log_{10} Q$')

# ---- twist number (row 2) ----
im2 = _plot_twist(data_muram_squashing_factor['twist'], panel_axs[2, 0])
im2 = _plot_twist(data_1slices_squashing_factor['twist'], panel_axs[2, 1])
im2 = _plot_twist(data_2slices_squashing_factor['twist'], panel_axs[2, 2])
im2 = _plot_twist(data_2slices_ambiguous_squashing_factor['twist'], panel_axs[2, 3])

fig.colorbar(im2, cax=axs[2, 4], location='right', label=r'Twist Number')

[ax.axhline(6.0, color='red', linestyle=':') for ax in panel_axs[0, :]]

[ax.set_xlim([-1, 9]) for ax in panel_axs.flatten()]
[ax.set_ylim([0, 10]) for ax in panel_axs.flatten()]

# remove x-tick labels for top 2 rows
[ax.set_xticklabels([]) for ax in panel_axs[:2, :].flatten()]
# remove y-tick labels for right 3 cols
[ax.set_yticklabels([]) for ax in panel_axs[:, 1:].flatten()]
# set x and y labels for leftmost col and bottom row
for i in range(3):
    panel_axs[i, 0].set_ylabel('Z [Mm]')
for j in range(4):
    panel_axs[2, j].set_xlabel('Y [Mm]')

fig.savefig(os.path.join(out_path, f'sma_{x_slice}.png'), dpi=300, transparent=True)
plt.close(fig)
