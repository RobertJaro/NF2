import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.unpack import load_cube

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_ideal'

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

b = load_cube(f'{base_path}/extrapolation_result.nf2')

# crop to same region
B = B[:, :, 20:80]  # apply offset
B = B[:, :, :b.shape[2]]  # crop to shape
# b = b[:, :, :B.shape[2]]  # crop to shape

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
