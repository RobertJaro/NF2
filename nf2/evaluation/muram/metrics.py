import numpy as np
from astropy.nddata import block_reduce

from nf2.evaluation.metric import evaluate
from nf2.evaluation.unpack import load_cube

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bxyz_v2'

dict_data = dict(np.load('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/Bvector.250000.npz'))
B = np.stack([dict_data['by'], dict_data['bz'], dict_data['bx']], -1) * np.sqrt(4 * np.pi)
B = np.moveaxis(B, 0, -2)
B = block_reduce(B, (2, 2, 6, 1), np.mean)  # reduce to HMI resolution

b = load_cube(f'{base_path}/extrapolation_result.nf2')

# crop to same region
B = B[:, :, 20:80]  # apply offset
b = b[:, :, :B.shape[2]]  # crop to shape

result = evaluate(b, B)

with open(f'{base_path}/evaluation.txt', 'w') as f:
    for k, v in result.items():
        print(k, f'{v:.02f}', file=f)
