import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.metric import evaluate, curl
from nf2.potential.potential_field import get_potential_field

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_comparison'

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution
B = B[:, :, 20:80]  # apply offset

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'))
b_slices = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_slices = np.moveaxis(b_slices, 0, -2)
b_slices = block_reduce(b_slices, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution

b = get_potential_field(b_slices[:, :, 0, 2], B.shape[2])

result = evaluate(b, B)

with open(f'{base_path}/evaluation.txt', 'w') as f:
    for k, v in result.items():
        print(k, f'{v:.02f}', file=f)
