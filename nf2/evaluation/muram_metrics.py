import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce

from nf2.data.analytical_field import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential
from nf2.train.metric import vector_norm, curl, divergence

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_extrapolation_v2'

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

b = load_cube(f'{base_path}/extrapolation_result.nf2')

# crop to same region
B = B[:, :, 20:80] # apply offset
b = b[:, :, :B.shape[2]] # crop to shape

heights = np.linspace(0, 1, 6) ** 2 * (B.shape[2] - 1)
heights = heights.astype(int)
fig, axs = plt.subplots(2, heights.shape[0], figsize=(12, 4))
[(ax.get_xaxis().set_ticks([]), ax.get_yaxis().set_ticks([])) for ax in np.ravel(axs)]
for i, h in enumerate(heights):
    v_min_max = np.abs(B[:, :, h, 2]).max()
    axs[0, i].imshow(B[:, :, h, 2].T, origin='lower', cmap='gray', vmin=-v_min_max, vmax=v_min_max)
    axs[1, i].imshow(b[:, :, h, 2].T, origin='lower', cmap='gray', vmin=-v_min_max, vmax=v_min_max)
    axs[0, i].set_title(f'{h * (0.192 * 2):.02f} Mm')


axs[0, 0].set_ylabel('MURaM', fontsize=20)
axs[1, 0].set_ylabel('PINN', fontsize=20)

fig.tight_layout()
fig.savefig(f'{base_path}/slice.jpg', dpi=300)
plt.close()


# plot
heights = np.mgrid[:B.shape[2]] * (0.192 * 2)

result = {}
result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
M = np.prod(B.shape[:-1])
result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

result['E_n'] = 1 - vector_norm(b - B).sum() / vector_norm(B).sum()

result['E_m'] = 1 - 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()

result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

B_potential = get_potential(B[:, :, 0, 2], 64)

result['eps_p'] = (vector_norm(b) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()
result['eps_p_ll'] = (vector_norm(B) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

j = curl(b)
result['sig_J'] = (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / vector_norm(j).sum() * 1e2
J = curl(B)
result['sig_J_ll'] = (vector_norm(np.cross(J, B, -1)) / vector_norm(B)).sum() / vector_norm(J).sum() * 1e2

result['L1'] = (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
result['L2'] = (divergence(b) ** 2).mean()

result['L1_B'] = (vector_norm(np.cross(curl(B), B, -1)) ** 2 / vector_norm(B) ** 2).mean()
result['L2_B'] = (divergence(B) ** 2).mean()

with open(f'{base_path}/evaluation.txt', 'w') as f:
    for k, v in result.items():
        print(k, f'{v:.02f}', file=f)