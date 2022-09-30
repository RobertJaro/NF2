import numpy as np

from nf2.evaluation.analytical_field import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential
from nf2.train.metric import vector_norm, curl, divergence

base_path = '/gpfs/gpfs0/robert.jarolim/nf2/analytical_caseW_v3'

# CASE 1
# B = get_analytic_b_field()
# CASE 2
B = get_analytic_b_field(n = 1, m = 1, l=0.3, psi=np.pi * 0.15, resolution=[80, 80, 72])
b = load_cube(f'{base_path}/extrapolation_result.nf2')

# for CASE 2 crop central 64^3
B = B[8:-8, 8:-8, :64]
b = b[8:-8, 8:-8, :64]

c_vec = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
M = np.prod(B.shape[:-1])
c_cs = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

E_n = vector_norm(b - B).sum() / vector_norm(B).sum()

E_m = 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()

eps = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

B_potential = get_potential(B[:, :, 0, 2], 64)

eps_p = (vector_norm(b) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()
eps_p_ll = (vector_norm(B) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

j = curl(b)
sig_J = (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / vector_norm(j).sum()

L1 = (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
L2 = (divergence(b) ** 2).mean()

L1_B = (vector_norm(np.cross(curl(B), B, -1)) ** 2 / vector_norm(B) ** 2).mean()
L2_B = (divergence(B) ** 2).mean()

with open(f'{base_path}/evaluation.txt', 'w') as f:
    print('c_vec', c_vec, file=f)
    print('c_cs', c_cs, file=f)
    print('E_n', 1 - E_n, file=f)
    print('E_m', 1 - E_m, file=f)
    print('eps', eps, file=f)
    print('eps_P', eps_p, file=f)
    print('eps_P_LL', eps_p_ll, file=f)
    print('sig_J * 1e2', sig_J * 1e2, file=f)
    print('L1', L1, file=f)
    print('L2', L2, file=f)
    print('L1_B', L1_B, file=f)
    print('L2_B', L2_B, file=f)