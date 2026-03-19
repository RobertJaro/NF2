import argparse
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

from nf2.evaluation.metric import curl, divergence, sigma_J, theta_J, vector_norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--n-workers', type=int, default=16)
    return parser.parse_args()


def compute_metrics(path):
    data = np.load(path, allow_pickle=True)
    b = np.asarray(data['b'])
    j = np.asarray(data['j']) if 'j' in data else curl(b)
    div_b = divergence(b)
    return (
        os.path.basename(path),
        np.nanmean(np.abs(div_b)),
        np.nanmean(np.abs(div_b) / (vector_norm(b) + 1e-7)),
        theta_J(b, j),
        sigma_J(b, j),
    )


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with mp.Pool(args.n_workers) as pool:
        rows = list(tqdm(pool.imap(compute_metrics, args.data), total=len(args.data)))
    widths = [
        max(len('filename'), *(len(row[0]) for row in rows)),
        len('mean_|divB|'),
        len('mean_|divB|/|B|'),
        len('theta_J_[deg]'),
        len('sigma_J'),
    ]
    lines = [
        f"{'filename':<{widths[0]}}  {'mean_|divB|':>{widths[1]}}  {'mean_|divB|/|B|':>{widths[2]}}  {'theta_J_[deg]':>{widths[3]}}  {'sigma_J':>{widths[4]}}",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}  {'-' * widths[4]}",
    ]
    for filename, mean_abs_div_b, mean_normalized_div_b, theta_j_deg, sigma_j in rows:
        lines.append(
            f"{filename:<{widths[0]}}  {mean_abs_div_b:>{widths[1]}.6e}  {mean_normalized_div_b:>{widths[2]}.6e}  {theta_j_deg:>{widths[3]}.6f}  {sigma_j:>{widths[4]}.6e}"
        )
    table = '\n'.join(lines)
    with open(args.output_file, 'w', encoding='ascii') as file:
        file.write(table + '\n')


if __name__ == '__main__':
    main()
