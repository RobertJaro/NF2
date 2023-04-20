import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce

from nf2.data.analytical_field import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential
from nf2.train.metric import vector_norm, curl, divergence

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_comparison'
os.makedirs(base_path, exist_ok=True)

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

# crop to same region
B = B[:, :, 20:80] # apply offset



# plot
heights = np.mgrid[:B.shape[2]] * (0.192 * 2)

fig, axs = plt.subplots(1, 3, figsize=(8, 4))

for path, label in zip(['/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_v2',
                        '/gpfs/gpfs0/robert.jarolim/multi_height/muram_fixed_v2',
                        '/gpfs/gpfs0/robert.jarolim/multi_height/muram_v1',
                        ], ['Extrapolation', 'Fixed Heights', 'Mapping']):
    b = load_cube(f'{path}/extrapolation_result.nf2')
    b = b[:, :, :B.shape[2]] # crop to shape
    #
    c_vec = np.sum((B * b).sum(-1), (0, 1)) / np.sqrt((B ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
    M = np.prod(B.shape[:-2])
    c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b), (0, 1))
    eps = (vector_norm(b) ** 2).sum((0, 1)) / (vector_norm(B) ** 2).sum((0, 1))
    #
    axs[0].plot(c_vec, heights, label=label)
    axs[1].plot(c_cs, heights, label=label)
    axs[2].plot(eps, heights, label=label)

axs[2].legend(loc='upper right')
axs[0].set_ylabel('Height [Mm]')

axs[0].set_xlabel('$c_{vec}$')
axs[1].set_xlabel('$c_{cs}$')
axs[2].set_xlabel('$\epsilon$')

axs[1].set_yticklabels([])
axs[2].set_yticklabels([])

axs[0].set_ylim(0, None)
axs[1].set_ylim(0, None)
axs[2].set_ylim(0, None)

fig.savefig(f'{base_path}/metrics.jpg')
plt.close(fig)
