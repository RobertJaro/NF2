
import numpy as np

from nf2.evaluation.analytical_solution import get_analytic_b_field
from nf2.evaluation.unpack import load_cube
from nf2.potential.potential_field import get_potential
from nf2.train.metric import vector_norm, curl

B = get_analytic_b_field()
b = load_cube('/gpfs/gpfs0/robert.jarolim/nf2/analytical_case1/extrapolation_result.nf2')

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

print('c_vec', c_vec)
print('c_cs', c_cs)
print('E_n', 1 - E_n)
print('E_m', 1 - E_m)
print('eps', eps)
print('eps_P', eps_p)
print('eps_P_LL', eps_p_ll)
print('sig_J * 1e2', sig_J * 1e2)
