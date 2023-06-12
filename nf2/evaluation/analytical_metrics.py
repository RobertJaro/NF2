import numpy as np

from nf2.data.analytical_field import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential, get_potential_field
from nf2.train.metric import vector_norm, curl, divergence

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/analytical_4tau_emb'

# CASE 1
B = get_analytic_b_field()
# CASE 2
# B = get_analytic_b_field(n = 1, m = 1, l=0.3, psi=np.pi * 0.15, resolution=[80, 80, 72])
b = load_cube(f'{base_path}/extrapolation_result.nf2')

# crop central 64^3
B = B[8:-8, 8:-8, :-8]
b = b[8:-8, 8:-8, :-8]

result = {}
result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
M = np.prod(B.shape[:-1])
result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

result['E_n'] = vector_norm(b - B).sum() / vector_norm(B).sum()

result['E_m'] = 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()

result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

B_potential = get_potential_field(B[:, :, 0, 2], 64)

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