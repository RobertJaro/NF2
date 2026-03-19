import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import LogNorm, Normalize

from nf2.evaluation.metric import b_nabla_bz
from nf2.evaluation.output_metrics import squashing_factor


def parse_args():
    parser = argparse.ArgumentParser(description='Plot a SHARP overview from one NPZ cube.')
    parser.add_argument('--data', required=True, help='Path to the NPZ file.')
    parser.add_argument('--output-dir', required=True, help='Directory where the overview plot is written.')
    parser.add_argument('--output-name', default=None, help='Output filename. Defaults to <npz_stem>_overview.png.')
    parser.add_argument('--x-slice', type=float, default=None, help='X slice position in Mm. Defaults to the cube center.')
    parser.add_argument('--x-lim', type=float, nargs=2, metavar=('X_MIN', 'X_MAX'), default=None)
    parser.add_argument('--y-lim', type=float, nargs=2, metavar=('Y_MIN', 'Y_MAX'), default=None)
    parser.add_argument('--z-lim', type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'), default=None)
    parser.add_argument('--q-vmax', type=float, default=1e3, help='Upper bound for squashing factor color scale.')
    return parser.parse_args()


def _to_dict(data):
    return {k: data[k] for k in data.files}


def _to_value(data, unit=None):
    if hasattr(data, 'to_value'):
        return data.to_value(unit) if unit is not None else data.value
    return np.asarray(data)


def _scalar_limits(coords):
    x = _to_value(coords[..., 0], u.Mm)
    y = _to_value(coords[..., 1], u.Mm)
    z = _to_value(coords[..., 2], u.Mm)
    return (
        float(np.nanmin(x)), float(np.nanmax(x)),
        float(np.nanmin(y)), float(np.nanmax(y)),
        float(np.nanmin(z)), float(np.nanmax(z)),
    )


def _infer_mm_per_pixel(data):
    if 'Mm_per_pixel' in data:
        return float(np.asarray(data['Mm_per_pixel']))

    coords = _to_value(data['coords'], u.Mm)
    steps = []
    for axis in range(3):
        axis_coords = coords[..., axis]
        diffs = np.diff(axis_coords, axis=axis)
        diffs = np.abs(np.asarray(diffs, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            steps.append(float(np.nanmedian(diffs)))
    if not steps:
        raise KeyError('Could not infer Mm_per_pixel from the NPZ file.')
    return min(steps)


def _prepare_metrics(data, x_slice_pix):
    cube_b = data['b']
    if 'b_nabla_bz' not in data:
        mm_per_pixel = _infer_mm_per_pixel(data)
        data['b_nabla_bz'] = b_nabla_bz(cube_b) / mm_per_pixel
    return squashing_factor(cube_b, x_range=[x_slice_pix, x_slice_pix + 1])


def _plot_bottom_bz(ax, b, extent, x_slice):
    im = ax.imshow(
        _to_value(b[:, :, 0, 2], u.G).T,
        origin='lower',
        cmap='gray',
        extent=extent,
        norm=Normalize(vmin=-2000, vmax=2000),
        aspect='auto',
    )
    ax.axvline(x_slice, color='red', linestyle='--', linewidth=1)
    return im


def _plot_b_nabla_bz(ax, data, x_slice_pix, extent):
    return ax.imshow(
        _to_value(data[x_slice_pix, :, :]).T,
        origin='lower',
        cmap='coolwarm',
        vmin=-0.1,
        vmax=0.1,
        extent=extent,
        aspect='auto',
    )


def _plot_q(ax, data, extent, q_vmax):
    return ax.imshow(
        _to_value(data[0, :, :]).T,
        origin='lower',
        cmap='viridis',
        extent=extent,
        norm=LogNorm(vmin=1, vmax=q_vmax),
        aspect='auto',
    )


def _plot_twist(ax, data, extent):
    return ax.imshow(
        _to_value(data[0, :, :]).T,
        origin='lower',
        cmap='seismic',
        extent=extent,
        vmin=-1,
        vmax=1,
        aspect='auto',
    )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = _to_dict(np.load(args.data, allow_pickle=True))
    coords = data['coords']
    x_min, x_max, y_min, y_max, z_min, z_max = _scalar_limits(coords)

    x_values = np.linspace(x_min, x_max, coords.shape[0])
    x_slice = args.x_slice if args.x_slice is not None else float(x_values[len(x_values) // 2])
    x_slice_pix = int(np.argmin(np.abs(x_values - x_slice)))

    squashing = _prepare_metrics(data, x_slice_pix)

    xy_extent = [x_min, x_max, y_min, y_max]
    yz_extent = [y_min, y_max, z_min, z_max]

    fig = plt.figure(figsize=(10.5, 6.8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1], hspace=0.32, wspace=0.22)

    ax_bz = fig.add_subplot(gs[0, :])
    ax_bnbz = fig.add_subplot(gs[1, 0])
    ax_q = fig.add_subplot(gs[1, 1])
    ax_twist = fig.add_subplot(gs[1, 2])

    im_bz = _plot_bottom_bz(ax_bz, data['b'], xy_extent, x_slice)
    im_bnbz = _plot_b_nabla_bz(ax_bnbz, data['b_nabla_bz'], x_slice_pix, yz_extent)
    im_q = _plot_q(ax_q, squashing['q'], yz_extent, args.q_vmax)
    im_twist = _plot_twist(ax_twist, squashing['twist'], yz_extent)

    cbar_bz = fig.colorbar(im_bz, ax=ax_bz, location='right', fraction=0.035, pad=0.02)
    cbar_bz.set_label(r'Bottom Boundary $B_z$ [G]')
    cbar_bnbz = fig.colorbar(im_bnbz, ax=ax_bnbz, location='top', fraction=0.07, pad=0.14)
    cbar_bnbz.set_label(r'$\hat{B} \cdot \nabla \hat{B}_z$ [1/Mm]')
    cbar_q = fig.colorbar(im_q, ax=ax_q, location='top', fraction=0.07, pad=0.14)
    cbar_q.set_label('Squashing Factor Q')
    cbar_twist = fig.colorbar(im_twist, ax=ax_twist, location='top', fraction=0.07, pad=0.14)
    cbar_twist.set_label('Twist Number')

    ax_bz.set_xlabel('X [Mm]')
    ax_bz.set_ylabel('Y [Mm]')
    ax_bz.set_title(f'x = {x_values[x_slice_pix]:.2f} Mm')

    for ax in (ax_bnbz, ax_q, ax_twist):
        ax.set_xlabel('Y [Mm]')
    ax_bnbz.set_ylabel('Z [Mm]')
    ax_q.set_yticklabels([])
    ax_twist.set_yticklabels([])

    if args.y_lim is not None:
        for ax in (ax_bnbz, ax_q, ax_twist):
            ax.set_xlim(args.y_lim)
    if args.z_lim is not None:
        for ax in (ax_bnbz, ax_q, ax_twist):
            ax.set_ylim(args.z_lim)

    output_name = args.output_name
    if output_name is None:
        stem = os.path.splitext(os.path.basename(args.data))[0]
        output_name = f'{stem}_overview.png'

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, output_name), dpi=300, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
