import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential_field
from nf2.evaluation.metric import vector_norm

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

avg_heights = np.array([0.960, 1.579, 3.882,  11.622, 19.378, 59.592]) * Mm_per_pix
[[ax.axhline(h, linestyle='--', color='black', alpha=0.5) for ax in axs] for h in avg_heights]
axs[-1].axvline(1, linestyle='--', color='black')

ckpt_paths = [
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_pf/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_fixed/extrapolation_result.nf2',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_ideal/extrapolation_result.nf2',
    # '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau/extrapolation_result.nf2',
    # '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bxyz_split/extrapolation_result.nf2',
    # '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bz_2epochs/extrapolation_result.nf2',
    # '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_pf/extrapolation_result.nf2',
]
labels = [
    'Extrapolation',
    'Extrapolation - PB',
    'Fixed Heights',
    'Mapped Heights',
    # r'Realistic vector',
    # r'Realistic split',
    # r'Realistic LOS',
    # 'Extrapolation - PB',
]

linestyle = [
            ':', ':',
             '--', '--',
             # 'dashdot', 'dashdot',
             # 'dashdot',
             # ':',
             ]

# assert same length
assert len(ckpt_paths) == len(labels) == len(linestyle)

colors = ['C1', 'C2', 'C3', 'C4']
# colors = ['C5', 'C6', 'C7', 'C2']

def _plot(b, label, c, ls):
    c_vec = np.sum((B * b).sum(-1), (0, 1)) / np.sqrt((B ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
    M = np.prod(B.shape[:-2])
    c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b), (0, 1))
    #
    E_n = 1 - vector_norm(b - B).sum((0, 1)) / vector_norm(B).sum((0, 1))
    E_m = 1 - 1 / M * (vector_norm(b - B) / vector_norm(B)).sum((0, 1))
    #
    eps = (vector_norm(b) ** 2).sum((0, 1)) / (vector_norm(B) ** 2).sum((0, 1))
    #
    axs[0].plot(c_vec, heights, label=label, color=c, linestyle=ls)
    axs[1].plot(c_cs, heights, label=label, color=c, linestyle=ls)
    axs[2].plot(E_n, heights, label=label, color=c, linestyle=ls)
    axs[3].plot(E_m, heights, label=label, color=c, linestyle=ls)
    axs[4].plot(eps, heights, label=label, color=c, linestyle=ls)

for path, label, c, ls in zip(ckpt_paths, labels, colors, linestyle):
    b = load_cube(path)
    b = b[:, :, :B.shape[2]]  # crop to shape
    #
    _plot(b, label, c, ls)

# plot potential field
b = get_potential_field(b_slices[:, :, 0, 2], B.shape[2])
_plot(b, 'Potential Field', 'C0', '-')

# plot legend centered at bottom of figure with 3 columns
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.), fancybox=True, shadow=True)

axs[0].set_ylabel('Height [Mm]')

axs[0].set_xlabel('$c_{vec}$')
axs[1].set_xlabel('$c_{cs}$')
axs[2].set_xlabel("$E_{n}'$")
axs[3].set_xlabel("$E_{m}'$")
axs[4].set_xlabel('$\epsilon$')

[ax.set_yticklabels([]) for ax in axs[1:]]

axs[0].set_xlim(0.7, 1)
axs[1].set_xlim(0.25, .85)
axs[2].set_xlim(0., .8)
axs[3].set_xlim(-0.5, .6)
axs[4].set_xlim(0.5, 1.5)

[ax.set_ylim(0, heights.max()) for ax in axs]


# fig.tight_layout()
fig.subplots_adjust(bottom=.27)
fig.savefig(f'{base_path}/comparison_metrics_full_v2.jpg', dpi=300)
plt.close(fig)
