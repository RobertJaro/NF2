import argparse
import os

import numpy as np
from astropy import constants, units as u
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import LogNorm, Normalize

from nf2.evaluation.metric import curl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-name', default='integrated_current_height_maps.png')
    parser.add_argument('--data-muram', required=True, help='Path to the reference MURaM npz file.')
    parser.add_argument('--data-slices', required=True, help='Path to the NF2 slices npz file.')
    parser.add_argument('--data-heights', required=True, help='Path to the NF2 height mapping file.')
    parser.add_argument('--tau-target', type=float, default=1e-6, help='Tau level used for the MURaM height map.')
    return parser.parse_args()


def _to_dict(npz_data):
    if isinstance(npz_data, np.lib.npyio.NpzFile):
        return {key: npz_data[key] for key in npz_data.files}
    return dict(npz_data)


def _get_mm_per_pixel(data):
    if 'Mm_per_pixel' not in data:
        raise KeyError('Missing Mm_per_pixel in input data.')
    return float(np.asarray(data['Mm_per_pixel']))


def _extract_extent(data, image_shape):
    coords = data.get('coords')
    if coords is not None:
        return [
            float(np.nanmin(coords[..., 0])),
            float(np.nanmax(coords[..., 0])),
            float(np.nanmin(coords[..., 1])),
            float(np.nanmax(coords[..., 1])),
        ]

    mm_per_pixel = _get_mm_per_pixel(data)
    return [0, image_shape[0] * mm_per_pixel, 0, image_shape[1] * mm_per_pixel]


def _compute_integrated_current_map(data):
    mm_per_pixel = _get_mm_per_pixel(data)
    m_per_pixel = mm_per_pixel * 1e6

    if 'j' in data:
        j = np.asarray(data['j'])
    elif 'b' in data:
        j = (curl(np.asarray(data['b'])) / mm_per_pixel * u.G / u.Mm) * constants.c / (4 * np.pi)
        j = j.to_value(u.G / u.s)
    else:
        raise KeyError('Input data needs either a j or b array.')

    return np.linalg.norm(j, axis=-1).sum(axis=2) * m_per_pixel


def _compute_muram_height_map(data, tau_target):
    if 'tau' not in data:
        raise KeyError('MURaM input data needs a tau array.')
    return np.argmin(np.abs(np.asarray(data['tau']) - tau_target), axis=2) * _get_mm_per_pixel(data)


def _height_entries(height_data):
    if isinstance(height_data, np.ndarray):
        if height_data.dtype == object:
            return list(height_data)
        return [height_data]

    if isinstance(height_data, np.lib.npyio.NpzFile):
        if len(height_data.files) == 1:
            entry = height_data[height_data.files[0]]
            return _height_entries(entry)
        return [height_data[key] for key in height_data.files]

    return [height_data]


def _compute_nf2_height_map(height_path):
    loaded = np.load(height_path, allow_pickle=True)
    entries = _height_entries(loaded)

    height_maps = []
    for entry in entries:
        if isinstance(entry, np.void) and entry.dtype.names is not None:
            coords = entry['coords']
        elif isinstance(entry, dict):
            coords = entry['coords']
        elif hasattr(entry, 'item'):
            item = entry.item()
            coords = item['coords'] if isinstance(item, dict) else item.coords
        else:
            coords = entry['coords']
        height_maps.append(np.asarray(coords)[..., 0, 2])

    if not height_maps:
        raise ValueError(f'Could not extract any height maps from {height_path}.')

    return np.mean(np.stack(height_maps, axis=0), axis=0)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_muram = _to_dict(np.load(args.data_muram))
    data_slices = _to_dict(np.load(args.data_slices))

    muram_current_map = _compute_integrated_current_map(data_muram)
    nf2_current_map = _compute_integrated_current_map(data_slices)

    muram_height_map = _compute_muram_height_map(data_muram, args.tau_target)
    nf2_height_map = _compute_nf2_height_map(args.data_heights)

    current_stack = np.concatenate([muram_current_map.ravel(), nf2_current_map.ravel()])
    current_stack = current_stack[np.isfinite(current_stack) & (current_stack > 0)]
    current_norm = LogNorm(vmin=current_stack.min(), vmax=current_stack.max())

    height_stack = np.concatenate([muram_height_map.ravel(), nf2_height_map.ravel()])
    height_stack = height_stack[np.isfinite(height_stack)]
    height_norm = Normalize(vmin=height_stack.min(), vmax=height_stack.max())

    muram_extent = _extract_extent(data_muram, muram_current_map.shape)
    nf2_extent = _extract_extent(data_slices, nf2_current_map.shape)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[1, 1, 0.06, 0.06],
        hspace=0.15, wspace=0.1
    )

    ax_muram_current = fig.add_subplot(gs[0, 0])
    ax_muram_height = fig.add_subplot(gs[0, 1])
    ax_nf2_current = fig.add_subplot(gs[1, 0])
    ax_nf2_height = fig.add_subplot(gs[1, 1])
    cax_current = fig.add_subplot(gs[2, :])
    cax_height = fig.add_subplot(gs[3, :])

    im_current = ax_muram_current.imshow(
        muram_current_map.T, origin='lower', cmap='plasma', extent=muram_extent, norm=current_norm
    )
    ax_muram_current.set_title('Integrated Current Density')
    ax_muram_current.set_ylabel('Y [Mm]')

    im_height = ax_muram_height.imshow(
        muram_height_map.T, origin='lower', cmap='viridis', extent=muram_extent, norm=height_norm
    )
    ax_muram_height.set_title('Height Map')

    ax_nf2_current.imshow(
        nf2_current_map.T, origin='lower', cmap='plasma', extent=nf2_extent, norm=current_norm
    )
    ax_nf2_current.set_ylabel('Y [Mm]')
    ax_nf2_current.set_xlabel('X [Mm]')

    ax_nf2_height.imshow(
        nf2_height_map.T, origin='lower', cmap='viridis', extent=nf2_extent, norm=height_norm
    )
    ax_nf2_height.set_xlabel('X [Mm]')

    fig.colorbar(im_current, cax=cax_current, orientation='horizontal',
                 label=r'$\sum_z |\mathbf{j}|$ [G$^2$ m / s]')
    fig.colorbar(im_height, cax=cax_height, orientation='horizontal', label='Height [Mm]')

    fig.text(0.03, 0.76, 'MURaM', rotation=90, va='center', ha='center')
    fig.text(0.03, 0.38, 'NF2', rotation=90, va='center', ha='center')
    fig.text(0.74, 0.535, fr'$\tau={args.tau_target:.0e}$', va='center', ha='center')

    for ax in [ax_muram_current, ax_muram_height]:
        ax.set_xticklabels([])
    for ax in [ax_muram_height, ax_nf2_height]:
        ax.set_yticklabels([])

    fig.savefig(os.path.join(args.output_dir, args.output_name), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
