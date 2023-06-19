import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nf2.evaluation.unpack import load_cube
from nf2.evaluation.metric import normalized_divergence, weighted_theta

base_path = '/gpfs/gpfs0/robert.jarolim/nf2/parameter_study'

evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

result = []
for nf2_file in tqdm(sorted(glob.glob(os.path.join(base_path, '**', '*.nf2')))):
    model = torch.load(nf2_file)['model']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    b = load_cube(nf2_file)
    div = normalized_divergence(b).mean()
    sig = weighted_theta(b).mean()
    #
    result += [(total_params, div, sig)]

result = np.array(result)

idx = np.argsort(result[:, 0])
result = result[idx, :]


result[:, 0] *= 1e-6

fig, axs = plt.subplots(2, 1, figsize=(5, 4))
axs[0].plot(result[:, 0], result[:, 1])
axs[1].plot(result[:, 0], result[:, 2])

t_ax1 = axs[0].twiny()
t_ax1.scatter(result[:, 0], result[:, 1], c='red' )
t_ax1.xaxis.set_ticks(result[:, 0], [16, '', '', '', 256, 512])
axs[0].set_xticklabels([])
t_ax1.set_xlabel('Nodes per Layer')

t_ax2 = axs[1].twiny()
t_ax2.scatter(result[:, 0], result[:, 2], c='red' )
t_ax2.xaxis.set_ticks(result[:, 0], ['','', '', '', '', ''])

axs[1].set_xlabel('Number of Parameters [1e6]')
axs[0].set_ylabel(r'$\langle |\nabla \cdot B| / |B| \rangle$ [G/pixel]')
axs[1].set_ylabel(r'$\theta_J$ [deg]')

axs[0].set_xticks([])
t_ax2.set_xticks([])

plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, 'parameter_variation.jpg'), dpi=300)
plt.close(fig)