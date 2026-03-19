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
    parser.add_argument('--data-muram', required=True)
    parser.add_argument('--data-1slices', required=True)
    parser.add_argument('--data-2slices', required=True)
    parser.add_argument('--data-2slices-ambiguous', required=True)
    parser.add_argument('--data-2slices-heights', required=True)
    parser.add_argument('--data-2slices-ambiguous-heights', required=True)
    parser.add_argument('--x-slice', type=float, default=-13)
    parser.add_argument('--y-lim', type=float, nargs=2, metavar=('Y_MIN', 'Y_MAX'), default=[2, 12])
    parser.add_argument('--z-lim', type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'), default=[0, 10])
    return parser.parse_args()


#############################################################
# create plot with 4 columns and 3 rows visualizing the following metrics:
# - b_nabla_bz
# - squashing factor Q
# - twist number
# add in heights of the 2-slices and ambiguous 2-slices as dashed black lines

def _plot_b_nabla_bz(data, ax, x_slice_pix, y_min, y_max, z_min, z_max, heights=None):
    im = ax.imshow(data[x_slice_pix, :, :].T,
                   origin='lower', cmap='coolwarm', vmin=-.1, vmax=.1, extent=[y_min, y_max, z_min, z_max])
    if heights is not None:
        for h in heights:
            ax.plot(np.linspace(y_min, y_max, h.shape[1]), h[x_slice_pix, :].to_value(u.Mm),
                    color='black', linestyle='--')
    return im


def _plot_squashing_factor_Q(data, ax, y_min, y_max, z_min, z_max, z_offset):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='viridis',
                   extent=[y_min, y_max, z_min + z_offset, z_max],
                   norm=LogNorm(vmin=1, vmax=1e3))
    return im


def _plot_twist(data, ax, y_min, y_max, z_min, z_max, z_offset):
    im = ax.imshow(data[0, :, :].T,
                   origin='lower', cmap='seismic',
                   extent=[y_min, y_max, z_min + z_offset, z_max], vmin=-1, vmax=1)
    return im


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_muram = dict(np.load(args.data_muram))
    data_1slices = np.load(args.data_1slices)
    data_2slices = np.load(args.data_2slices)
    data_2slices_ambiguous = np.load(args.data_2slices_ambiguous)

    data_2slices_heights = np.load(args.data_2slices_heights, allow_pickle=True)
    data_2slices_ambiguous_heights = np.load(args.data_2slices_ambiguous_heights, allow_pickle=True)

    coords = data_1slices['coords']  # (x, y, z, 3)
    x_min, x_max = coords[:, :, :, 0].min(), coords[:, :, :, 0].max()
    y_min, y_max = coords[:, :, :, 1].min(), coords[:, :, :, 1].max()
    z_min, z_max = coords[:, :, :, 2].min(), coords[:, :, :, 2].max()

    x_slice_pix = np.argmin(np.abs(np.linspace(x_min, x_max, coords.shape[0]) - args.x_slice))

    offset = int(1.5 / data_muram['Mm_per_pixel'])  # Mm -> pixels
    z_offset = offset * data_muram['Mm_per_pixel']
    data_muram_squashing_factor = squashing_factor(data_muram['b'][:, :, offset:], x_range=[x_slice_pix, x_slice_pix + 1])
    data_1slices_squashing_factor = squashing_factor(data_1slices['b'][:, :, offset:], x_range=[x_slice_pix, x_slice_pix + 1])
    data_2slices_squashing_factor = squashing_factor(data_2slices['b'][:, :, offset:], x_range=[x_slice_pix, x_slice_pix + 1])
    data_2slices_ambiguous_squashing_factor = squashing_factor(
        data_2slices_ambiguous['b'][:, :, offset:], x_range=[x_slice_pix, x_slice_pix + 1]
    )

    muram_heights = []
    muram_tau = data_muram['tau']
    for tau in [1e-6]:
        height_muram = np.argmin(np.abs(muram_tau - tau), axis=2) * data_muram['Mm_per_pixel'] * u.Mm
        muram_heights.append(height_muram)

    data_2slices_heights = [d['coords'][:, :, 0, 2] for d in data_2slices_heights]
    data_2slices_ambiguous_heights = [d['coords'][:, :, 0, 2] for d in data_2slices_ambiguous_heights]

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(
        3, 5,
        width_ratios=[1, 1, 1, 1, 0.05],
        wspace=0.15, hspace=0.15
    )

    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(3)])
    panel_axs = axs[:, :4]
    cax0 = axs[0, 4]
    cax1 = axs[1, 4]

    im0 = _plot_b_nabla_bz(data_muram['b_nabla_bz'], panel_axs[0, 0], x_slice_pix, y_min, y_max, z_min, z_max, heights=muram_heights)
    im0 = _plot_b_nabla_bz(data_1slices['b_nabla_bz'], panel_axs[0, 1], x_slice_pix, y_min, y_max, z_min, z_max)
    im0 = _plot_b_nabla_bz(
        data_2slices['b_nabla_bz'], panel_axs[0, 2], x_slice_pix, y_min, y_max, z_min, z_max, heights=data_2slices_heights
    )
    im0 = _plot_b_nabla_bz(
        data_2slices_ambiguous['b_nabla_bz'], panel_axs[0, 3], x_slice_pix, y_min, y_max, z_min, z_max,
        heights=data_2slices_ambiguous_heights
    )
    fig.colorbar(im0, cax=cax0, location='right', label=r'$\hat{B} \cdot \nabla \hat{B}_z$')

    im1 = _plot_squashing_factor_Q(data_muram_squashing_factor['q'], panel_axs[1, 0], y_min, y_max, z_min, z_max, z_offset)
    im1 = _plot_squashing_factor_Q(data_1slices_squashing_factor['q'], panel_axs[1, 1], y_min, y_max, z_min, z_max, z_offset)
    im1 = _plot_squashing_factor_Q(data_2slices_squashing_factor['q'], panel_axs[1, 2], y_min, y_max, z_min, z_max, z_offset)
    im1 = _plot_squashing_factor_Q(
        data_2slices_ambiguous_squashing_factor['q'], panel_axs[1, 3], y_min, y_max, z_min, z_max, z_offset
    )
    fig.colorbar(im1, cax=cax1, location='right', label=r'$Q$')

    im2 = _plot_twist(data_muram_squashing_factor['twist'], panel_axs[2, 0], y_min, y_max, z_min, z_max, z_offset)
    im2 = _plot_twist(data_1slices_squashing_factor['twist'], panel_axs[2, 1], y_min, y_max, z_min, z_max, z_offset)
    im2 = _plot_twist(data_2slices_squashing_factor['twist'], panel_axs[2, 2], y_min, y_max, z_min, z_max, z_offset)
    im2 = _plot_twist(
        data_2slices_ambiguous_squashing_factor['twist'], panel_axs[2, 3], y_min, y_max, z_min, z_max, z_offset
    )
    fig.colorbar(im2, cax=axs[2, 4], location='right', label=r'$T$')
    [ax.axhline(6.0, color='red', linestyle=':') for ax in panel_axs[0, :]]
    [ax.set_xlim(args.y_lim) for ax in panel_axs.flatten()]
    [ax.set_ylim(args.z_lim) for ax in panel_axs.flatten()]
    [ax.set_xticklabels([]) for ax in panel_axs[:2, :].flatten()]
    [ax.set_yticklabels([]) for ax in panel_axs[:, 1:].flatten()]

    for i in range(3):
        panel_axs[i, 0].set_ylabel('Z [Mm]')
    for j in range(4):
        panel_axs[2, j].set_xlabel('Y [Mm]')

    fig.savefig(os.path.join(args.output_dir, f'{args.x_slice}.png'), dpi=300, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
