import argparse

from nf2.evaluation.metric import normalized_divergence, weighted_theta, theta_J, sigma_J
from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.output_metrics import current_density


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')

    args = parser.parse_args()
    nf2_path = args.nf2_path

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(progress=True, metrics={'j': current_density})

    metrics = {}
    b = output['b']
    j = output['j']
    metrics['normalized_divergence'] = normalized_divergence(b)
    metrics['theta_J'] = theta_J(b, j)
    metrics['sigma_J'] = sigma_J(b, j)

    print(f'Normalized divergence: {metrics["normalized_divergence"]:.3f}')
    print(f'Theta_J: {metrics["theta_J"]:.3f}')
    print(f'Sigma_J (1e2): {metrics["sigma_J"] * 1e2:.3f}')


if __name__ == '__main__':
    main()