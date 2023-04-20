import os

import numpy as np
import torch
from matplotlib import pyplot as plt


base_path = '/gpfs/gpfs0/robert.jarolim/nf2/377_benchmarking'
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

state = torch.load(os.path.join(base_path, 'checkpoint.pt'))

history = state['history']

fig, axs = plt.subplots(3, 1, figsize=(4, 4), sharex=True)
iterations = (np.array(history['iteration']) + 1) * 1e-4
axs[0].plot(iterations, history['b_loss'])
axs[1].plot(iterations, history['divergence_loss'])
axs[2].plot(iterations, history['force_loss'])

axs[2].set_xlabel('Iterations [1e4]')

axs[0].set_ylabel(r'$L_{B0}$')
axs[1].set_ylabel(r'$L_{div}$')
axs[2].set_ylabel(r'$L_{ff}$')

axs[0].set_ylim(0, None)
axs[1].set_ylim(0, None)
axs[2].set_ylim(0, None)

axs[0].axvline(5, linestyle='--', c='red')
axs[1].axvline(5, linestyle='--', c='red')
axs[2].axvline(5, linestyle='--', c='red')

axs[0].axvline(8, linestyle='--', c='black')
axs[1].axvline(8, linestyle='--', c='black')
axs[2].axvline(8, linestyle='--', c='black')

axs[0].axvline(10, linestyle='--', c='black')
axs[1].axvline(10, linestyle='--', c='black')
axs[2].axvline(10, linestyle='--', c='black')

plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, 'history.jpg'), dpi=300)
plt.close(fig)