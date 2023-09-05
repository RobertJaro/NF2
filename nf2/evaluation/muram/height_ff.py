import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential_field
from nf2.evaluation.metric import vector_norm, curl, divergence

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_comparison'
os.makedirs(base_path, exist_ok=True)

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'))
b_slices = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_slices = np.moveaxis(b_slices, 0, -2)
b_slices = block_reduce(b_slices, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution

# crop to same region
B = B[:, :, 20:90]  # apply offset

# plot
Mm_per_pix = (0.192 * 2)
heights = np.mgrid[:B.shape[2]] * Mm_per_pix

fig, axs = plt.subplots(1, 5, figsize=(12, 4))
[ax.set_axis_off() for ax in axs[2:]]

avg_heights = np.array([0.960, 1.579, 3.882,  11.622, 19.378, 59.592]) * Mm_per_pix
[[ax.axhline(h, linestyle='--', color='black', alpha=0.5) for ax in axs[:2]] for h in avg_heights]

ckpt_paths = [
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_pf/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_fixed/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_ideal/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bxyz_split/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bz_2epochs/extrapolation_result.nf2',
]
labels = [
    'Extrapolation',
    'Extrapolation - PB',
    'Fixed Heights',
    'Mapped Heights',
    r'Realistic vector',
    r'Realistic split',
    r'Realistic LOS',
]

linestyle = [':', ':',
             '--', '--',
             'dashdot', 'dashdot', 'dashdot',]

# assert same length
assert len(ckpt_paths) == len(labels) == len(linestyle)

colors = [f'C{i + 1}' for i in range(len(ckpt_paths))]

def _plot(b, label, c, ls):
    j = curl(b)
    sig = (vector_norm(np.cross(j, b)) / vector_norm(B)).sum((0, 1)) / vector_norm(j).sum((0, 1))
    #
    L_div = (np.sqrt(divergence(b) ** 2) / vector_norm(b)).mean((0, 1))
    #
    axs[0].plot(sig, heights, label=label, color=c, linestyle=ls)
    axs[1].plot(L_div, heights, label=label, color=c, linestyle=ls)

for path, label, c, ls in zip(ckpt_paths, labels, colors, linestyle):
    b = load_cube(path)
    b = b[:, :, :B.shape[2]]  # crop to shape
    #
    _plot(b, label, c, ls)

# plot potential field
# b = get_potential_field(b_slices[:, :, 0, 2], B.shape[2])
# _plot(b, 'Potential Field', 'C0', '-')

axs[0].plot(np.NaN, np.NaN, '-', color='none', label=' ') # placeholder for legend
_plot(B, 'MURaM', 'C9', '-')

# dummy potential label
axs[0].plot([], [], label='Potential', color="C0", linestyle='-')

axs[0].plot(np.NaN, np.NaN, '-', color='none', label='  ') # placeholder for legend

# plot legend centered at bottom of figure with 3 columns
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.67, 0.5), fancybox=True, shadow=True, prop={'size': 12})

axs[0].set_ylabel('Height [Mm]')

axs[0].set_xlabel(r'$\sigma_J$')
axs[1].set_xlabel(r'$L_{\rm div, n}$')

[ax.set_yticklabels([]) for ax in axs[1:]]

axs[0].set_xlim(0., 0.4)
axs[1].set_xlim(0., .02)
# axs[2].set_xlim(0., .8)
# axs[3].set_xlim(-0.5, .5)
# axs[4].set_xlim(0.5, 1.5)

[ax.set_ylim(0, heights.max()) for ax in axs]


# fig.tight_layout()
fig.subplots_adjust(bottom=.27)
fig.savefig(f'{base_path}/comparison_ff.jpg', dpi=300)
plt.close(fig)
