import argparse

from nf2.data.analytical_field import get_analytic_b_field
from nf2.evaluation.metric import evaluate
from nf2.evaluation.unpack import load_cube

# argparser
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('base_path', type=str)
args = parser.parse_args()

base_path = args.base_path

# CASE 1
B = get_analytic_b_field()
# CASE 2
# B = get_analytic_b_field(n = 1, m = 1, l=0.3, psi=np.pi * 0.15, resolution=[80, 80, 72])

b = load_cube(f'{base_path}/extrapolation_result.nf2')

# crop central 64^3
B = B[8:-8, 8:-8, :-8]
b = b[8:-8, 8:-8, :-8]
result = evaluate(b, B)

with open(f'{base_path}/evaluation.txt', 'w') as f:
    for k, v in result.items():
        print(k, f'{v:.02f}', file=f)
