import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import LogNorm

from nf2.evaluation.output_metrics import squashing_factor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-1slices', required=True)
    parser.add_argument('--data-2slices', required=True)
    parser.add_argument(
        '--data-2slices-heights',
        required=True,
    )
    parser.add_argument('--x-slice', type=float, default=-60)
    parser.add_argument('--y-lim', type=float, nargs=2, metavar=('Y_MIN', 'Y_MAX'), default=[-15, 15])
    parser.add_argument('--z-lim', type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'), default=[0, 15])
    return parser.parse_args()


#############################################################
def _plot_b_nabla_bz(data, ax, heights, x_slice_pix, y_min, y_max, z_min, z_max):
    im = ax.imshow(data[x_slice_pix, :, :].T,
                   origin='lower', cmap='coolwarm', vmin=-.1, vmax=.1, extent=[y_min, y_max, z_min, z_max])
    if heights is not None:
        for h in heights:
            ax.plot(np.linspace(y_min, y_max, h.shape[1]), h[x_slice_pix, :].to_value(u.Mm),
                    color='black', linestyle='--')
    return im


def _plot_squashing_factor_Q(data, ax, y_min, y_max, z_min, z_max):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='viridis',
                   extent=[y_min, y_max, z_min, z_max],
                   norm=LogNorm(vmin=1, vmax=1e3))
    return im


def _plot_twist(data, ax, y_min, y_max, z_min, z_max):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='seismic',
                   extent=[y_min, y_max, z_min, z_max], vmin=-1, vmax=1)
    return im


def _to_current_density_values(data):
    current_density = np.linalg.norm(data, axis=-1)
    if hasattr(current_density, 'to_value'):
        return current_density.to_value(u.G / u.s)
    return current_density


def _plot_current_density(data, ax, x_slice_pix, y_min, y_max, z_min, z_max):
    im = ax.imshow(
        data[x_slice_pix, :, :].T,
        origin='lower',
        cmap='inferno',
        extent=[y_min, y_max, z_min, z_max],
        norm=LogNorm(vmin=1, vmax=1e3),
    )
    return im


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_1slices = np.load(args.data_1slices)
    data_2slices = dict(np.load(args.data_2slices))

    data_2slices['b'] = data_2slices['b'][:, :, :data_1slices['b'].shape[2]]
    data_2slices['b_nabla_bz'] = data_2slices['b_nabla_bz'][:, :, :data_1slices['b_nabla_bz'].shape[2]]
    data_2slices['j'] = data_2slices['j'][:, :, :data_1slices['j'].shape[2]]

    data_2slices_heights = np.load(args.data_2slices_heights, allow_pickle=True)
    data_2slices_heights = [d['coords'][:, :, 0, 2] for d in data_2slices_heights]

    coords = data_1slices['coords']  # (x, y, z, 3)
    x_min, x_max = coords[:, :, :, 0].min(), coords[:, :, :, 0].max()
    y_min, y_max = coords[:, :, :, 1].min(), coords[:, :, :, 1].max()
    z_min, z_max = coords[:, :, :, 2].min(), coords[:, :, :, 2].max()

    # get pixel index close to x_slice in Mm
    x_slice_pix = np.argmin(np.abs(np.linspace(x_min, x_max, coords.shape[0]) - args.x_slice))

    data_1slices_squashing_factor = squashing_factor(data_1slices['b'], x_range=[x_slice_pix, x_slice_pix + 1])
    data_2slices_squashing_factor = squashing_factor(data_2slices['b'], x_range=[x_slice_pix, x_slice_pix + 1])
    data_1slices_current_density = _to_current_density_values(data_1slices['j'])
    data_2slices_current_density = _to_current_density_values(data_2slices['j'])

    fig = plt.figure(figsize=(10.5, 3.5))
    gs = gridspec.GridSpec(
        3, 4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[0.05, 1, 1],
        wspace=0.15, hspace=0.15
    )

    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)])
    panel_axs = axs[1:, :]

    im0 = _plot_b_nabla_bz(data_1slices['b_nabla_bz'], panel_axs[0, 0], None, x_slice_pix, y_min, y_max, z_min, z_max)
    im0 = _plot_b_nabla_bz(
        data_2slices['b_nabla_bz'], panel_axs[1, 0], data_2slices_heights, x_slice_pix, y_min, y_max, z_min, z_max
    )
    fig.colorbar(im0, cax=axs[0, 0], location='top', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [1/Mm]')

    im1 = _plot_squashing_factor_Q(data_1slices_squashing_factor['q'], panel_axs[0, 1], y_min, y_max, z_min, z_max)
    im1 = _plot_squashing_factor_Q(data_2slices_squashing_factor['q'], panel_axs[1, 1], y_min, y_max, z_min, z_max)
    fig.colorbar(im1, cax=axs[0, 1], location='top', label=r'Squashing Factor Q')

    im2 = _plot_twist(data_1slices_squashing_factor['twist'], panel_axs[0, 2], y_min, y_max, z_min, z_max)
    im2 = _plot_twist(data_2slices_squashing_factor['twist'], panel_axs[1, 2], y_min, y_max, z_min, z_max)
    fig.colorbar(im2, cax=axs[0, 2], location='top', label=r'Twist Number')

    im3 = _plot_current_density(data_1slices_current_density, panel_axs[0, 3], x_slice_pix, y_min, y_max, z_min, z_max)
    im3 = _plot_current_density(data_2slices_current_density, panel_axs[1, 3], x_slice_pix, y_min, y_max, z_min, z_max)
    fig.colorbar(im3, cax=axs[0, 3], location='top', label=r'Current Density $|J|$ [G/s]')

    [ax.set_xlim(args.y_lim) for ax in panel_axs.flatten()]
    [ax.set_ylim(args.z_lim) for ax in panel_axs.flatten()]
    [ax.set_xticklabels([]) for ax in panel_axs[:1, :].flatten()]
    [ax.set_yticklabels([]) for ax in panel_axs[:, 1:].flatten()]

    for i in range(2):
        panel_axs[i, 0].set_ylabel('Z [Mm]')

    for j in range(4):
        panel_axs[1, j].set_xlabel('Y [Mm]')

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'sst_pre_{args.x_slice}.png'), dpi=300, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
