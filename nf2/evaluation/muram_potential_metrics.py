import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.data.analytical_field import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential, get_potential_field
from nf2.train.metric import vector_norm, curl, divergence, energy

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_comparison'

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'))
b_slices = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_slices = np.moveaxis(b_slices, 0, -2)
b_slices = block_reduce(b_slices, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution


b = get_potential_field(b_slices[:, :, 0, 2], B.shape[2])

# crop to same region
B = B[:, :, 20:80] # apply offset
b = b[:, :, :B.shape[2]] # crop to shape

result = {}
result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
M = np.prod(B.shape[:-1])
result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

result['E_n'] = 1 - vector_norm(b - B).sum() / vector_norm(B).sum()

result['E_m'] = 1 - 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()

result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

B_potential = get_potential_field(B[:, :, 0, 2], B.shape[2])

result['eps_free'] = (energy(b) - energy(B_potential)).sum() / (energy(B) - energy(B_potential)).sum()

result['eps_p'] = (vector_norm(b) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()
result['eps_p_ll'] = (vector_norm(B) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

# c = 29979245800
j = curl(b) #/ (192e5 * 2) * c / (4 * np.pi) # gauss / pixel --> gauss / cm
result['sig_J'] = (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / vector_norm(j).sum() * 1e2
J = curl(B) #/ (192e5 * 2) * c / (4 * np.pi) # gauss / pixel --> gauss / cm
result['sig_J_ll'] = (vector_norm(np.cross(J, B, -1)) / vector_norm(B)).sum() / vector_norm(J).sum() * 1e2

result['L1'] = (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
result['L2'] = (divergence(b) ** 2).mean()

result['L1_B'] = (vector_norm(np.cross(curl(B), B, -1)) ** 2 / vector_norm(B) ** 2).mean()
result['L2_B'] = (divergence(B) ** 2).mean()

with open(f'{base_path}/potential_evaluation.txt', 'w') as f:
    for k, v in result.items():
        print(k, f'{v:.02f}', file=f)