import argparse
import os
import pickle
import tempfile
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from nf2.evaluation.metric import theta_J
from nf2.loader.muram import MURaMDataset


DEFAULT_DATA_DIR = '/glade/work/rjarolim/nf2/topology/npzs'
DEFAULT_MURAM_SLICE_DIR = '/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/2D'


@dataclass
class Dataset:
    label: str
    lambda_label: str
    path: str
    n_boundaries: int
    height_path: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot global quality metrics for MURaM lambda_ff topology runs.'
    )
    parser.add_argument(
        '--output-file',
        default='/glade/work/rjarolim/nf2/topology/results/curvature_mfr/muram_mfr_lambda_quality_metrics.png',
    )
    parser.add_argument(
        '--boundary-tau-1',
        default=os.path.join(DEFAULT_MURAM_SLICE_DIR, 'tau_slice_1.000.474000'),
    )
    parser.add_argument(
        '--boundary-tau-1e-6',
        default=os.path.join(DEFAULT_MURAM_SLICE_DIR, 'tau_slice_0.000001.474000'),
    )
    parser.add_argument('--data-1slices-ff1e-3', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-3_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-3', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-3_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-3-heights', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-3_height_v01.pkl'))
    parser.add_argument('--data-1slices-ff5e-4', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff5.0e-4_v01.npz'))
    parser.add_argument('--data-2slices-ff5e-4', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-4_v01.npz'))
    parser.add_argument('--data-2slices-ff5e-4-heights', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-4_height_v01.pkl'))
    parser.add_argument('--data-1slices-ff1e-4', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-4_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-4', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-4_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-4-heights', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-4_height_v01.pkl'))
    parser.add_argument('--data-1slices-ff5e-5', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff5.0e-5_v01.npz'))
    parser.add_argument('--data-2slices-ff5e-5', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-5_v01.npz'))
    parser.add_argument('--data-2slices-ff5e-5-heights', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-5_height_v01.pkl'))
    parser.add_argument('--data-1slices-ff1e-5', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-5_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-5', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-5_v01.npz'))
    parser.add_argument('--data-2slices-ff1e-5-heights', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-5_height_v01.pkl'))
    return parser.parse_args()


@lru_cache(maxsize=None)
def _load_muram_boundary(path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = MURaMDataset(
            path,
            los_trv_azi_transform=False,
            batch_size=2 ** 20,
            work_directory=tmp_dir,
            shuffle=False,
            plot=False,
        )
        boundary = np.load(dataset.file_paths['b_true'], mmap_mode='r').reshape((*dataset.cube_shape, 3))
        boundary = np.asarray(boundary) * 2500
        dataset.clear()
    return boundary


def _boundary_specs(boundary_tau_1, boundary_tau_1e_6):
    return [
        {'ds_id': 'tau_1.0', 'data_path': boundary_tau_1},
        {'ds_id': 'tau_1.0e-6', 'data_path': boundary_tau_1e_6},
    ]


def _load_height_layer_fields(path):
    with open(path, 'rb') as file:
        height_data = pickle.load(file)
    fields = []
    for entry in height_data:
        if 'b' not in entry:
            raise KeyError(
                f'Missing magnetic field b in {path}. Re-run nf2_height_to_npz after updating the converter.'
            )
        field = np.asarray(entry['b'])
        if field.ndim == 4 and field.shape[2] == 1:
            field = field[:, :, 0]
        fields.append(field)
    return fields


def _surface_error(model_boundary, boundary):
    x_count = min(model_boundary.shape[0], boundary.shape[0])
    y_count = min(model_boundary.shape[1], boundary.shape[1])
    delta = np.linalg.norm(model_boundary[:x_count, :y_count] - boundary[:x_count, :y_count], axis=-1)
    reference = np.linalg.norm(boundary[:x_count, :y_count], axis=-1)
    valid = np.isfinite(delta) & np.isfinite(reference) & (reference > 0)
    return np.nanmean(delta[valid])


def _relative_boundary_error(data, boundary_specs, dataset):
    b = np.asarray(data['b'])
    height_layer_fields = _load_height_layer_fields(dataset.height_path) if dataset.height_path is not None else []

    surface_errors = []
    for boundary_idx, spec in enumerate(boundary_specs[:dataset.n_boundaries]):
        boundary = _load_muram_boundary(spec['data_path'])
        if boundary_idx == 0:
            model_boundary = b[:, :, 0]
        else:
            model_boundary = height_layer_fields[boundary_idx - 1]
        surface_errors.append(_surface_error(model_boundary, boundary))

    return float(np.nanmean(surface_errors))


def _theta_j(data):
    b = np.asarray(data['b'])
    j = np.asarray(data['j']) if 'j' in data else None
    return float(theta_J(b, j))


def _datasets(args):
    return [
        Dataset('single-height', r'$\lambda_{\rm ff}=1 \times 10^{-3}$', args.data_1slices_ff1e_3, 1),
        Dataset('multi-height', r'$\lambda_{\rm ff}=1 \times 10^{-3}$', args.data_2slices_ff1e_3, 2, args.data_2slices_ff1e_3_heights),
        Dataset('single-height', r'$\lambda_{\rm ff}=5 \times 10^{-4}$', args.data_1slices_ff5e_4, 1),
        Dataset('multi-height', r'$\lambda_{\rm ff}=5 \times 10^{-4}$', args.data_2slices_ff5e_4, 2, args.data_2slices_ff5e_4_heights),
        Dataset('single-height', r'$\lambda_{\rm ff}=1 \times 10^{-4}$', args.data_1slices_ff1e_4, 1),
        Dataset('multi-height', r'$\lambda_{\rm ff}=1 \times 10^{-4}$', args.data_2slices_ff1e_4, 2, args.data_2slices_ff1e_4_heights),
        Dataset('single-height', r'$\lambda_{\rm ff}=5 \times 10^{-5}$', args.data_1slices_ff5e_5, 1),
        Dataset('multi-height', r'$\lambda_{\rm ff}=5 \times 10^{-5}$', args.data_2slices_ff5e_5, 2, args.data_2slices_ff5e_5_heights),
        Dataset('single-height', r'$\lambda_{\rm ff}=1 \times 10^{-5}$', args.data_1slices_ff1e_5, 1),
        Dataset('multi-height', r'$\lambda_{\rm ff}=1 \times 10^{-5}$', args.data_2slices_ff1e_5, 2, args.data_2slices_ff1e_5_heights),
    ]


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    boundary_specs = _boundary_specs(args.boundary_tau_1, args.boundary_tau_1e_6)
    datasets = _datasets(args)

    markers = {'single-height': 'o', 'multi-height': 's'}
    colors = {
        r'$\lambda_{\rm ff}=1 \times 10^{-3}$': 'tab:blue',
        r'$\lambda_{\rm ff}=5 \times 10^{-4}$': 'tab:orange',
        r'$\lambda_{\rm ff}=1 \times 10^{-4}$': 'tab:green',
        r'$\lambda_{\rm ff}=5 \times 10^{-5}$': 'tab:purple',
        r'$\lambda_{\rm ff}=1 \times 10^{-5}$': 'tab:red',
    }

    rows = []
    for dataset in datasets:
        data = np.load(dataset.path, allow_pickle=True)
        rows.append(
            {
                'dataset': dataset,
                'boundary_error': _relative_boundary_error(data, boundary_specs, dataset),
                'theta_j': _theta_j(data),
            }
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for row in rows:
        dataset = row['dataset']
        label_separator = ',  ' if dataset.label == 'multi-height' else ', '
        ax.scatter(
            row['boundary_error'],
            row['theta_j'],
            marker=markers[dataset.label],
            color=colors[dataset.lambda_label],
            s=55,
            edgecolor='black',
            linewidth=0.5,
            label=f'{dataset.label}{label_separator}{dataset.lambda_label}',
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=8,
        loc='upper right',
        bbox_to_anchor=(1.40, 1.05),
        frameon=True,
        shadow=True,
    )
    ax.set_xlabel(r'$\langle |\Delta B| \rangle$ [G]')
    ax.set_ylabel(r'$\theta_J$ [deg]')
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_file, dpi=300, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
