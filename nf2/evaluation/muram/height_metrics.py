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
B = B[:, :, 20:80]  # apply offset

# plot
heights = np.mgrid[:B.shape[2]] * (0.192 * 2)

fig, axs = plt.subplots(1, 5, figsize=(12, 4))

[ax.axhline(0.6, linestyle='--', color='black', alpha=0.5) for ax in axs]
[ax.axhline(1.0, linestyle='--', color='black', alpha=0.5) for ax in axs]
[ax.axhline(1.8, linestyle='--', color='black', alpha=0.5) for ax in axs]
[ax.axhline(2.7, linestyle='--', color='black', alpha=0.5) for ax in axs]
[ax.axhline(4.9, linestyle='--', color='black', alpha=0.5) for ax in axs]
axs[-1].axvline(1, linestyle='--', color='black')

# plot potential field
label = 'Potential Field'
c = 'C0'
b = get_potential_field(b_slices[:, :, 0, 2], B.shape[2])
c_vec = np.sum((B * b).sum(-1), (0, 1)) / np.sqrt((B ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
M = np.prod(B.shape[:-2])
c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b), (0, 1))
#
E_n = 1 - vector_norm(b - B).sum((0, 1)) / vector_norm(B).sum((0, 1))
E_m = 1 - 1 / M * (vector_norm(b - B) / vector_norm(B)).sum((0, 1))
#
eps = (vector_norm(b) ** 2).sum((0, 1)) / (vector_norm(B) ** 2).sum((0, 1))
#
axs[0].plot(c_vec, heights, label=label, color=c)
axs[1].plot(c_cs, heights, label=label, color=c)
axs[2].plot(E_n, heights, label=label, color=c)
axs[3].plot(E_m, heights, label=label, color=c)
axs[4].plot(eps, heights, label=label, color=c)

for path, label, c in zip([
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_pf',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_fixed',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_ideal',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bz_v3',
    '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bz_v4',
                           ], [
    'Extrapolation',
    'Fixed Heights',
    'Mapped Heights (ideal)',
    r'Mapped Heights ($\tau = 10^{-4}$)',
    r'Mapped Heights ($\tau = 10^{-4}$, $B_z$ only)',
    r'Mapped Heights ($\tau = 10^{-4}$, $B_z$ only, initialized)',
                               ],
                          ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8','C9']):
    # for path, label, c in zip(['/gpfs/gpfs0/robert.jarolim/multi_height/muram_l1e-0',
    #                         '/gpfs/gpfs0/robert.jarolim/multi_height/muram_l1e-1',
    #                         '/gpfs/gpfs0/robert.jarolim/multi_height/muram_l1e-2',
    #                         '/gpfs/gpfs0/robert.jarolim/multi_height/muram_l1e-3'
    #                         ], [r'$\lambda = 1$', r'$\lambda = 10^{-1}$', r'$\lambda = 10^{-2}$', r'$\lambda = 10^{-3}$'],
    #                           ['C2', 'red', 'C4', 'C5']):
    b = load_cube(f'{path}/extrapolation_result.nf2')
    b = b[:, :, :B.shape[2]]  # crop to shape
    #
    c_vec = np.sum((B * b).sum(-1), (0, 1)) / np.sqrt((B ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
    M = np.prod(B.shape[:-2])
    c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b), (0, 1))
    #
    E_n = 1 - vector_norm(b - B).sum((0, 1)) / vector_norm(B).sum((0, 1))
    E_m = 1 - 1 / M * (vector_norm(b - B) / vector_norm(B)).sum((0, 1))
    #
    eps = (vector_norm(b) ** 2).sum((0, 1)) / (vector_norm(B) ** 2).sum((0, 1))
    #
    axs[0].plot(c_vec, heights, label=label, color=c)
    axs[1].plot(c_cs, heights, label=label, color=c)
    axs[2].plot(E_n, heights, label=label, color=c)
    axs[3].plot(E_m, heights, label=label, color=c)
    axs[4].plot(eps, heights, label=label, color=c)

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

axs[0].set_xlim(0.6, 1)
axs[1].set_xlim(0.2, 1)
axs[2].set_xlim(0., 1)
axs[3].set_xlim(-0.5, 1)
axs[4].set_xlim(0.5, 1.5)

[ax.set_ylim(0, heights.max()) for ax in axs]


# fig.tight_layout()
fig.subplots_adjust(bottom=.27)
fig.savefig(f'{base_path}/comparison_metrics.jpg', dpi=300)
plt.close(fig)
