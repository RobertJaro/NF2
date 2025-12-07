import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import b_nabla_bz, energy
from nf2.evaluation.output import CartesianOutput, HeightTransformOutput
from nf2.evaluation.output_metrics import squashing_factor, free_energy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file.')
    parser.add_argument('--out_path', type=str, help='path to the output folder.', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.192 for MURaM)', required=False,
                        default=0.192)
    parser.add_argument('--height', type=float, help='height in Mm', required=False, default=10.0)
    parser.add_argument('--x_slice', type=float, help='x slice in Mm', required=False, default=7.0)
    parser.add_argument('--z_line', type=float, help='z line in Mm', required=False, default=None)
    parser.add_argument('--xlim', type=float, nargs=2, help='x limits in Mm', required=False, default=[2, 12])
    args = parser.parse_args()

    nf2_path = args.nf2_path
    out_path = args.out_path if args.out_path is not None else os.path.join(os.path.dirname(nf2_path),
                                                                            'evaluation_plots')
    os.makedirs(out_path, exist_ok=True)

    height = args.height
    x_slice = args.x_slice
    z_line = args.z_line

    nf2_model = CartesianOutput(nf2_path)
    nf2_out = nf2_model.load_cube([0, height], metrics=['j', 'b_nabla_bz'], progress=True,
                                  Mm_per_pixel=args.Mm_per_pixel)
    try:
        height_model = HeightTransformOutput(nf2_path)
        height_out = height_model.load_height_mapping()
    except Exception as e:
        print(f'Could not load height mapping: {e}')
        height_out = []

    Mm_per_ds = nf2_model.Mm_per_ds
    Mm_per_pixel = nf2_model.Mm_per_pixel
    ds_per_pixel = Mm_per_pixel / Mm_per_ds

    x_min, x_max = nf2_model.coord_range[0] * nf2_model.Mm_per_ds
    y_min, y_max = nf2_model.coord_range[1] * nf2_model.Mm_per_ds
    z_min, z_max = 0, height

    nf2_out['metrics']['b_nabla_bz'] = b_nabla_bz(nf2_out['b']) / Mm_per_pixel
    b = nf2_out['b']

    # compute twist and squashing factor with 2 Mm offset
    offset = int(2 / Mm_per_pixel)
    nf2_squashing_factor = squashing_factor(nf2_out['b'][:, :, offset:])

    # compute free magnetic energy
    nf2_free_energy = free_energy(nf2_out['b'])  # erg cm^-3

    # x_slice = 33 + x_min  # Mm
    # x_slice = 37 + x_min  # Mm
    # x_slice = 40 + x_min  # Mm
    xlim = args.xlim

    # get pixel index close to x_slice in Mm
    x_slice_pix = np.argmin(np.abs(np.linspace(x_min, x_max, b.shape[0]) - x_slice))
    print(f'x_slice_pix: {x_slice_pix}, x_slice: {x_slice} Mm')

    coords = nf2_out['coords']
    heights = [height_out[i]['coords'][:, :, 0, 2] for i in range(len(height_out))]

    ##############################################################################
    ######################### plot Bz
    xy_extent = [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    im = ax.imshow(b[:, :, 0, 2].to_value(u.G).T, origin='lower', cmap='gray', vmin=-1000, vmax=1000, extent=xy_extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')

    # draw red line at slice position
    ax.axvline(x_slice, color='red', linestyle='--')

    plt.savefig(os.path.join(out_path, 'bz.png'), dpi=300, transparent=True)
    plt.close()

    ##############################################################################
    ######################### plot b_nabla_bz

    yz_extent = [y_min, y_max, z_min, z_max]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    im = ax.imshow(nf2_out['metrics']['b_nabla_bz'][x_slice_pix, :, :].T,
                   origin='lower', cmap='coolwarm', vmin=-.1, vmax=.1, extent=yz_extent)

    for h_coords in heights:
        ax.plot(np.linspace(y_min, y_max, h_coords.shape[1]), h_coords[x_slice_pix, :].to_value(u.Mm),
                color='black', linestyle='--')

    # c = coords[x_slice_pix, :, :, 1:]
    # byz_pre = b[x_slice_pix, :, :, 1:]
    # coord_q = c  # [::2, ::2]  # block_reduce(coord, (3, 3, 1), np.mean)
    # b_q = byz_pre  # [::2, ::2]  # block_reduce(byz_pre, (3, 3, 1), np.mean)
    # b_q = b_q / np.linalg.norm(b_q, axis=-1, keepdims=True)
    # ax.quiver(coord_q[..., 0], coord_q[..., 1], b_q[..., 0], b_q[..., 1], color='darkgray', scale=40,
    #           pivot='middle')

    ax.set_ylim([0, height])
    # horizontal line at z=5 Mm
    if z_line is not None:
        ax.axhline(z_line, color='red', linestyle=':')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')

    ax.set_xlabel('Y [Mm]')
    ax.set_ylabel('Z [Mm]')

    ax.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'b_nabla_bz.png'), dpi=300, transparent=True)
    plt.close(fig)

    ##############################################################################
    ######################### plot energy

    extent = [y_min, y_max, 0, height]
    b_norm = LogNorm(vmin=1e2, vmax=1e5)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(energy(nf2_out['b'][x_slice_pix, :, :].to_value(u.G)).T, origin='lower', extent=extent,
              norm=b_norm, cmap='jet')

    ax.set_xlabel('Y [Mm]')
    ax.set_ylabel('Z [Mm]')

    ax.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'energy.png'), dpi=300, transparent=True)
    plt.close(fig)

    ##############################################################################
    ######################### plot j
    z_height = int(2 / Mm_per_pixel)

    extent = [x_min, x_max, y_min, y_max]
    j_norm = LogNorm()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    integrated_j = np.linalg.norm(nf2_out['metrics']['j'][:, :, z_height:].to_value(u.G / u.s), axis=-1).sum(
        2) * Mm_per_pixel * 1e8
    im = ax.imshow(integrated_j.T, origin='lower', extent=extent, norm=j_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'Integrated |J| [G cm s$^{-1}$]')

    ax.set_xlabel('X [Mm]')
    ax.set_ylabel('Y [Mm]')

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'j.png'), dpi=300, transparent=True)
    plt.close(fig)

    ##############################################################################
    ######################### plot twist

    twist_norm = Normalize(-2, 2)

    extent = [y_min, y_max, offset * Mm_per_pixel, height]

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    im = ax.imshow(nf2_squashing_factor['twist'][x_slice_pix, :, :].T, origin='lower', extent=extent,
                   norm=twist_norm,
                   cmap='seismic')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'Twist Number')

    ax.set_xlabel('Y [Mm]')
    ax.set_ylabel('Z [Mm]')

    ax.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'twist.png'), dpi=300, transparent=True)
    plt.close(fig)

    ##############################################################################
    ######################### plot squashing factor

    squashing_norm = LogNorm(vmin=1, vmax=1e3)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    im = ax.imshow(nf2_squashing_factor['q'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=squashing_norm,
                   cmap='viridis')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'Squashing Factor Q')

    ax.set_xlabel('Y [Mm]')
    ax.set_ylabel('Z [Mm]')

    ax.set_xlim(xlim)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'squashing_factor.png'), dpi=300,
                transparent=True)
    plt.close()
