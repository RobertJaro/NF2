import argparse
import os
import pickle
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from matplotlib import pyplot as plt

from nf2.evaluation.metric import theta_J
from nf2.loader.muram import MURaMDataset


DEFAULT_DATA_DIR = '/glade/work/rjarolim/nf2/topology/npzs'
DEFAULT_MURAM_SLICE_DIR = '/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/2D'


@dataclass
class Dataset:
    label: str
    path: str
    height_path: str


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot theta_J over DeltaB for SST positional-encoding ablation runs.'
    )
    parser.add_argument(
        '--output-file',
        default='/glade/work/rjarolim/nf2/topology/results/encoding_ablation/encoding_quality_metrics.png',
    )
    parser.add_argument(
        '--boundary-tau-1',
        default=os.path.join(DEFAULT_MURAM_SLICE_DIR, 'tau_slice_1.000.474000'),
    )
    parser.add_argument(
        '--boundary-tau-1e-6',
        default=os.path.join(DEFAULT_MURAM_SLICE_DIR, 'tau_slice_0.000001.474000'),
    )
    parser.add_argument('--data-encoding-16', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_16_v01.npz'))
    parser.add_argument('--data-encoding-16-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_16_height_v01.pkl'))
    parser.add_argument('--data-encoding-1', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_1_v01.npz'))
    parser.add_argument('--data-encoding-1-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_1_height_v01.pkl'))
    parser.add_argument('--data-encoding-2', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_2_v01.npz'))
    parser.add_argument('--data-encoding-2-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_2_height_v01.pkl'))
    parser.add_argument('--data-encoding-32', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_32_v01.npz'))
    parser.add_argument('--data-encoding-32-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_32_height_v01.pkl'))
    parser.add_argument('--data-encoding-4', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_4_v01.npz'))
    parser.add_argument('--data-encoding-4-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_4_height_v01.pkl'))
    parser.add_argument('--data-encoding-none', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_none_v01.npz'))
    parser.add_argument('--data-encoding-none-heights', default=os.path.join(DEFAULT_DATA_DIR, 'encoding_none_height_v01.pkl'))
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


def _boundary_error(data, boundary_specs, dataset):
    b = np.asarray(data['b'])
    height_layer_fields = _load_height_layer_fields(dataset.height_path)

    surface_errors = []
    for boundary_idx, spec in enumerate(boundary_specs):
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
        Dataset('None', args.data_encoding_none, args.data_encoding_none_heights),
        Dataset('1', args.data_encoding_1, args.data_encoding_1_heights),
        Dataset('2', args.data_encoding_2, args.data_encoding_2_heights),
        Dataset('4', args.data_encoding_4, args.data_encoding_4_heights),
        Dataset('16', args.data_encoding_16, args.data_encoding_16_heights),
        Dataset('32', args.data_encoding_32, args.data_encoding_32_heights),
    ]


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    boundary_specs = _boundary_specs(args.boundary_tau_1, args.boundary_tau_1e_6)

    rows = []
    for dataset in _datasets(args):
        data = np.load(dataset.path, allow_pickle=True)
        rows.append(
            {
                'dataset': dataset,
                'boundary_error': _boundary_error(data, boundary_specs, dataset),
                'theta_j': _theta_j(data),
            }
        )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
    for row, color in zip(rows, colors):
        dataset = row['dataset']
        ax.scatter(
            row['boundary_error'],
            row['theta_j'],
            s=60,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            label=dataset.label,
        )
        ax.annotate(
            dataset.label,
            (row['boundary_error'], row['theta_j']),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=9,
        )

    ax.legend(
        title='Encoding',
        fontsize=9,
        title_fontsize=10,
        loc='upper right',
        bbox_to_anchor=(1.08, 0.92),
        frameon=True,
        shadow=True,
    )
    ax.set_xlabel(r'$\langle |\Delta B| \rangle$ [G]')
    ax.set_ylabel(r'$\theta_J$ [deg]')
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_file, dpi=300, transparent=True)
    plt.close(fig)

    for row in rows:
        print(f"{row['dataset'].label}: DeltaB={row['boundary_error']:.6f} G, theta_J={row['theta_j']:.6f} deg")


if __name__ == '__main__':
    main()
