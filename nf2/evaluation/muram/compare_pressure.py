import argparse
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.output import CartesianOutput

from nf2.loader.muram import MURaMSnapshot
from astropy import units as u

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=0.192 * 2)
    parser.add_argument('--muram_source_path', type=str, help='path to the MURaM simulation.')
    parser.add_argument('--muram_iteration', type=int, help='iteration of the snapshot.')

    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    Mm_per_pixel = args.Mm_per_pixel

    snapshot = MURaMSnapshot(args.muram_source_path, args.muram_iteration)
    muram_P = snapshot.P[:, :, 64:]
    muram_B = snapshot.B[:, :, 64:]
    muram_dx_Mm_per_pixel = snapshot.ds[0].to_value(u.Mm / u.pix)
    muram_dz_Mm_per_pixel = snapshot.ds[2].to_value(u.Mm / u.pix)

    nf2_model = CartesianOutput(args.nf2_path)
    nf2_out = nf2_model.load_cube(Mm_per_pixel=Mm_per_pixel, metrics=['j'], progress=True)

    ########################################################
    # Plot comparison of photospheric pressure
    bottom_p_nf2 = nf2_out['p'][:, :, 0, 0].to_value(u.G ** 2) #nf2_out['p'].sum((2, -1)).to_value(u.G ** 2) * Mm_per_pixel
    bottom_p_muram = muram_P[:, :, 0] #muram_P.sum(2) * snapshot.ds[2].to_value(u.Mm / u.pix)

    norm = LogNorm(vmin=bottom_p_muram.min(), vmax=bottom_p_muram.max())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    im = ax.imshow(bottom_p_nf2.T, cmap='plasma', origin='lower', norm=norm)
    ax.set_title('NF2')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax = axs[1]
    im = ax.imshow(bottom_p_muram.T, cmap='plasma', origin='lower', norm=norm)
    ax.set_title('MURaM')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


    plt.savefig(os.path.join(args.out_path, 'bottom_pressure.png'))
    plt.close(fig)

    ########################################################
    # Plot initial pressure
    integrated_p_nf2 = nf2_out['p'].sum((2, -1)).to_value(u.G ** 2) * Mm_per_pixel
    integrated_p_muram = muram_P.sum(2) * snapshot.ds[2].to_value(u.Mm / u.pix)

    norm = LogNorm(vmin=integrated_p_muram.min(), vmax=integrated_p_muram.max())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    im = ax.imshow(integrated_p_nf2.T, cmap='plasma', origin='lower', norm=norm)
    ax.set_title('NF2')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax = axs[1]
    im = ax.imshow(integrated_p_muram.T, cmap='plasma', origin='lower', norm=norm)
    ax.set_title('MURaM')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.savefig(os.path.join(args.out_path, 'integrated_pressure.png'))
    plt.close(fig)

    ########################################################
    # Plot pressure slices
    Mm_range = np.linspace(0, 5, 6)
    fig, axs = plt.subplots(6, 2, figsize=(10, 10))

    for i in range(6):
        pix_nf2 = int(Mm_range[i] / Mm_per_pixel)
        nf2_pressure_slice = nf2_out['p'][:, :, pix_nf2, 0].to_value(u.G ** 2)
        pix_muram = int(Mm_range[i] / muram_dz_Mm_per_pixel)
        muram_pressure_slice = muram_P[:, :, pix_muram]
        #
        norm = LogNorm(vmin=muram_pressure_slice.min(), vmax=muram_pressure_slice.max())
        #
        ax = axs[i, 0]
        im = ax.imshow(nf2_pressure_slice.T, cmap='plasma', origin='lower', norm=norm)
        ax.set_title(f'NF2 - {Mm_range[i]} Mm')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        #
        ax = axs[i, 1]
        im = ax.imshow(muram_pressure_slice.T, cmap='plasma', origin='lower', norm=norm)
        ax.set_title(f'MURaM - {Mm_range[i]} Mm')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path, 'pressure_slices.png'))
    plt.close(fig)

    ########################################################
    # Plot pressure profile
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(muram_P.sum((0, 1)) * muram_dx_Mm_per_pixel ** 2, np.arange(muram_P.shape[2]) * muram_dz_Mm_per_pixel, label='MURaM - MEAN')
    ax.plot(nf2_out['p'].sum((0, 1, -1)) * Mm_per_pixel ** 2, np.arange(nf2_out['p'].shape[2]) * Mm_per_pixel, label='NF2 - MEAN')

    ax.set_ylabel('Height [Mm]')
    ax.set_xlabel('Pressure [erg/cm^3]')
    ax.semilogx()
    ax.legend()
    plt.savefig(os.path.join(args.out_path, 'pressure_profile.png'))
    plt.close(fig)

    ########################################################
    # Plot magnetic energy
    magnetic_energy_muram = (muram_B ** 2).sum(-1) / (8 * np.pi)
    magnetic_energy_nf2 = (nf2_out['b'] ** 2).sum(-1) / (8 * np.pi)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(magnetic_energy_nf2.sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2,
            np.arange(magnetic_energy_nf2.shape[2]) * (Mm_per_pixel * 1e8), label='NF2')
    ax.plot(magnetic_energy_muram.sum((0, 1)) * (muram_dx_Mm_per_pixel * 1e8) ** 2,
            np.arange(magnetic_energy_muram.shape[2]) * (muram_dz_Mm_per_pixel * 1e8), label='MURaM')

    ax.set_ylabel('Height [Mm]')
    ax.set_xlabel('Magnetic energy [erg/cm]')
    ax.semilogx()
    ax.legend()
    plt.savefig(os.path.join(args.out_path, 'magnetic_energy.png'))
    plt.close(fig)
