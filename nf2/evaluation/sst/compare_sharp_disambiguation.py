import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nf2.loader.fits import process_map
from sunpy.map import Map

from nf2.evaluation.output import CartesianOutput, DisambiguationOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
    parser.add_argument('--out_path', type=str, help='output path.')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    sharp_model = CartesianOutput('/glade/work/rjarolim/nf2/sst/sharp_13392_v01/extrapolation_result.nf2')
    amb_model = CartesianOutput('/glade/work/rjarolim/nf2/sst/sharp_13392_ambiguous_v01/extrapolation_result.nf2')

    Mm_per_pixel = 0.72
    sharp_out = sharp_model.load_cube(height_range=[0, 20], Mm_per_pixel=Mm_per_pixel, metrics=['j'])
    amb_out = amb_model.load_cube(height_range=[0, 20], Mm_per_pixel=Mm_per_pixel, metrics=['j'])

    b_sharp = sharp_out['b'][:, :, 0].to_value(u.G)
    b_amb = amb_out['b'][:, :, 0].to_value(u.G)

    xy_extent = [0, sharp_out['b'].shape[0] * Mm_per_pixel,
                 0, sharp_out['b'].shape[1] * Mm_per_pixel]


    fig, axs = plt.subplots(3, 3, figsize=(8, 6))

    for i, label in enumerate(['$B_x$', '$B_y$', '$B_z']):
        ax = axs[i, 0]
        disamb_im = ax.imshow(b_sharp[..., i].T, origin='lower', cmap='gray', extent=xy_extent, vmin=-1000, vmax=1000)

        ax = axs[i, 1]
        amb_im = ax.imshow(b_amb[..., i].T, origin='lower', cmap='gray', extent=xy_extent, vmin=-1000, vmax=1000)

        ax = axs[i, 2]
        diff = b_sharp[..., i] - b_amb[..., i]
        diff_im = ax.imshow(diff.T, origin='lower', cmap='RdBu', extent=xy_extent, vmin=-100, vmax=100)


    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(disamb_im, cax=cax, orientation='horizontal', label=r'$B_\text{disamb}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(amb_im, cax=cax, orientation='horizontal', label=r'$B_\text{amb}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(diff_im, cax=cax, orientation='horizontal', label=r'$B_\text{disamb} - B_\text{amb}$ [G]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    [ax.set_xlabel('X [Mm]') for ax in axs[-1, :]]
    [ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'sharp_disambiguation_comparison.png'), dpi=300, transparent=True)
    plt.close()

    ##############################
    # Plot current density

    cm_per_pixel = Mm_per_pixel * 1e8
    j_sharp = np.linalg.norm(sharp_out['metrics']['j'], axis=-1).to_value(u.G / u.s).sum(2) * cm_per_pixel
    j_amb = np.linalg.norm(amb_out['metrics']['j'], axis=-1).to_value(u.G / u.s).sum(2) * cm_per_pixel

    j_norm = LogNorm()

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    ax = axs[0]
    disamb_im = ax.imshow(j_sharp.T, origin='lower', cmap='inferno', extent=xy_extent, norm=j_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(disamb_im, cax=cax, orientation='horizontal', label=r'$|J_\text{disamb}|$ [G cm s$^{-1}$]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    ax = axs[1]
    amb_im = ax.imshow(j_amb.T, origin='lower', cmap='inferno', extent=xy_extent, norm=j_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(amb_im, cax=cax, orientation='horizontal', label=r'$|J_\text{amb}|$ [G cm s$^{-1}$]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    ax = axs[2]
    diff = (j_sharp - j_amb)* 1e-11
    v_min_max = np.max(np.abs(diff))
    diff_im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r', extent=xy_extent, vmin=-v_min_max, vmax=v_min_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(diff_im, cax=cax, orientation='horizontal', label=r'$|J_\text{disamb}| - |J_\text{amb}|$ [$10^{11}$ G cm s$^{-1}$]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    [ax.set_xlabel('X [Mm]') for ax in axs]
    axs[0].set_ylabel('Y [Mm]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'current_density_comparison.png'), dpi=300, transparent=True)
    plt.close()


    ##############################
    # Plot disambiguation maps

    sharp_prob = Map("/glade/work/rjarolim/data/nf2/13392/hmi.sharp_720s.9875.20230806_091200_TAI.conf_disambig.fits")
    sharp_prob.data[:] = np.flip(sharp_prob.data, axis=(0, 1))
    sharp_prob = process_map(sharp_prob, [380, 840, 0, 1000], 1)
    sharp_flip_prob = sharp_prob.data

    sharp_disamb = Map("/glade/work/rjarolim/data/nf2/13392/hmi.sharp_720s.9875.20230806_091200_TAI.disambig.fits")
    sharp_disamb.data[:] = np.flip(sharp_disamb.data, axis=(0, 1))
    sharp_disamb = process_map(sharp_disamb, [380, 840, 0, 1000], 1)

    sharp_field = Map("/glade/work/rjarolim/data/nf2/13392/hmi.sharp_720s.9875.20230806_091200_TAI.field.fits")
    sharp_field.data[:] = np.flip(sharp_field.data, axis=(0, 1))
    sharp_field = process_map(sharp_field, [380, 840, 0, 1000], 1)

    sharp_inc = Map("/glade/work/rjarolim/data/nf2/13392/hmi.sharp_720s.9875.20230806_091200_TAI.inclination.fits")
    sharp_inc.data[:] = np.flip(sharp_inc.data, axis=(0, 1))
    sharp_inc = process_map(sharp_inc, [380, 840, 0, 1000], 1)

    B_los = sharp_field.data * np.cos(np.deg2rad(sharp_inc.data))

    amb = sharp_disamb.data
    amb_weak = 2
    sharp_flip = (amb.astype(int) >> amb_weak).astype(bool)

    sharp_flip_prob[~sharp_flip] = 100 - sharp_flip_prob[~sharp_flip]

    nf2_model = DisambiguationOutput('/glade/work/rjarolim/nf2/sst/sharp_13392_ambiguous_v01/extrapolation_result.nf2')
    nf2_disamb_out = nf2_model.load_slice(0 * u.Mm)
    nf2_flip_prob = nf2_disamb_out[0]['flip'] * 100


    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    ax = axs[0]
    im = ax.imshow(sharp_flip_prob, origin='lower', cmap='RdBu_r', extent=xy_extent, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='horizontal', label='SHARP Flip Probability [%]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    ax = axs[1]
    im = ax.imshow(nf2_flip_prob[..., 0].T, origin='lower', cmap='RdBu_r', extent=xy_extent, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='horizontal', label='NF2 Flip Probability [%]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Br contours at -1000, 1000 G
    [ax.contour(B_los, levels=[-500, 500], colors=['darkgray', 'white'], extent=xy_extent, linewidths=1) for ax in axs]

    [ax.set_xlabel('X [Mm]') for ax in axs]
    [ax.set_ylabel('Y [Mm]') for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'disambiguation_comparison.png'), dpi=300, transparent=True)
    plt.close()
