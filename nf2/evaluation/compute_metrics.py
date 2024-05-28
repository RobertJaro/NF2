import argparse

from nf2.evaluation.metric import normalized_divergence, weighted_theta, sigma_J, theta_J
from nf2.evaluation.unpack import load_cube


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')

    args = parser.parse_args()
    nf2_path = args.nf2_path

    b = load_cube(nf2_path)

    metrics = {}
    metrics['normalized_divergence'] = normalized_divergence(b).mean()
    metrics['theta_J'] = theta_J(b)
    metrics['sigma_J'] = sigma_J(b)

    print(weighted_theta(b) - theta_J(b))

    print(f'Normalized divergence: {metrics["normalized_divergence"]:.3f}')
    print(f'Theta_J: {metrics["theta_J"]:.3f}')
    print(f'Sigma_J (1e2): {metrics["sigma_J"] * 1e2:.3f}')


if __name__ == '__main__':
    main()