import argparse
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.output_metrics import free_energy
from nf2.loader.muram import MURaMSnapshot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path_magnetostatic', type=str,
                        help='path to the source NF2 files for the magnetostatic model')
    parser.add_argument('--nf2_path_force_free', type=str, help='path to the source NF2 files for the force-free model')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=0.192 * 4)
    parser.add_argument('--muram_source_path', type=str, help='path to the MURaM simulation.')
    parser.add_argument('--muram_iteration', type=int, help='iteration of the snapshot.')

    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    Mm_per_pixel = args.Mm_per_pixel

    snapshot = MURaMSnapshot(args.muram_source_path, args.muram_iteration)
    muram_out = snapshot.load_cube(resolution=args.Mm_per_pixel * u.Mm / u.pix, target_tau=1.0, height=60 * u.Mm)
    # muram_P = snapshot.P[:, :, 116:]
    # muram_B = snapshot.B[:, :, 116:]
    muram_B = muram_out['B']
    muram_P = muram_out['P']
    muram_dx_Mm_per_pixel = args.Mm_per_pixel  # snapshot.ds[0].to_value(u.Mm / u.pix)
    muram_dz_Mm_per_pixel = args.Mm_per_pixel  # snapshot.ds[2].to_value(u.Mm / u.pix)

    nf2_magnetostatic__model = CartesianOutput(args.nf2_path_magnetostatic)
    nf2_magnetostatic_outputs = nf2_magnetostatic__model.load_cube(Mm_per_pixel=Mm_per_pixel, metrics=['j'],
                                                                   progress=True, height_range=[0, 60])

    nf2_force_free_model = CartesianOutput(args.nf2_path_force_free)
    nf2_force_free_outputs = nf2_force_free_model.load_cube(Mm_per_pixel=Mm_per_pixel, metrics=['j'], progress=True)

    ########################################################
    # Plot magnetic energy
    magnetic_energy_muram = (muram_B ** 2).sum(-1) / (8 * np.pi)
    magnetic_energy_nf2_magnetostatic = (nf2_magnetostatic_outputs['b'] ** 2).sum(-1) / (8 * np.pi)
    magnetic_energy_nf2_force_free = (nf2_force_free_outputs['b'] ** 2).sum(-1) / (8 * np.pi)

    free_energy_muram = free_energy(muram_B * u.G)['free_energy']
    free_energy_nf2_magnetostatic = free_energy(nf2_magnetostatic_outputs['b'])['free_energy']
    free_energy_nf2_force_free = free_energy(nf2_force_free_outputs['b'])['free_energy']

    fig, axs = plt.subplots(1, 2, figsize=(5, 5))

    ax = axs[0]
    ax.plot(magnetic_energy_nf2_magnetostatic.sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2,
            np.arange(magnetic_energy_nf2_magnetostatic.shape[2]) * (Mm_per_pixel * 1e8), label='NF2 magnetostatic',
            linestyle='dashed')
    ax.plot(magnetic_energy_nf2_force_free.sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2,
            np.arange(magnetic_energy_nf2_force_free.shape[2]) * (Mm_per_pixel * 1e8), label='NF2 force-free',
            linestyle='dotted')
    ax.plot(magnetic_energy_muram.sum((0, 1)) * (muram_dx_Mm_per_pixel * 1e8) ** 2,
            np.arange(magnetic_energy_muram.shape[2]) * (muram_dz_Mm_per_pixel * 1e8), label='MURaM')
    ax.set_xlabel('Magnetic energy [erg/cm]')

    ax = axs[1]
    ax.plot(free_energy_nf2_magnetostatic.sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2,
            np.arange(free_energy_nf2_magnetostatic.shape[2]) * (Mm_per_pixel * 1e8), label='NF2 magnetostatic',
            linestyle='dashed')
    ax.plot(free_energy_nf2_force_free.sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2,
            np.arange(free_energy_nf2_force_free.shape[2]) * (Mm_per_pixel * 1e8), label='NF2 force-free',
            linestyle='dotted')
    ax.plot(free_energy_muram.sum((0, 1)) * (muram_dx_Mm_per_pixel * 1e8) ** 2,
            np.arange(free_energy_muram.shape[2]) * (muram_dz_Mm_per_pixel * 1e8), label='MURaM')
    ax.set_xlabel('Free energy [erg/cm]')

    axs[0].set_ylabel('Height [Mm]')
    axs[-1].legend()

    # [ax.semilogx() for ax in axs]
    plt.savefig(os.path.join(args.out_path, 'magnetic_energy.png'))
    plt.close(fig)


    ########################################################
    # Plot difference metrics

    def compute_metrics(b, B):
        M = np.prod(B.shape[:-2])
        #
        E_n = 1 - np.linalg.norm(b - B, axis=-1).sum((0, 1)) / np.linalg.norm(B, axis=-1).sum((0, 1))
        E_m = 1 - 1 / M * (np.linalg.norm(b - B, axis=-1) / np.linalg.norm(B, axis=-1)).sum((0, 1))
        c_vec = np.sum((B * b).sum(-1), (0, 1)) / np.sqrt((B ** 2).sum(-1).sum((0, 1)) * (b ** 2).sum(-1).sum((0, 1)))
        c_cs = 1 / M * np.sum((B * b).sum(-1) / np.linalg.norm(B, axis=-1) / np.linalg.norm(b, axis=-1), (0, 1))
        eps = (np.linalg.norm(b, axis=-1) ** 2).sum((0, 1)) / (np.linalg.norm(B, axis=-1) ** 2).sum((0, 1))
        return {'E_n': E_n, 'c_cs': c_cs, 'c_vec': c_vec, 'E_m': E_m, 'eps': eps}

    nf2_ff_b = nf2_force_free_outputs['b'].to_value(u.G)
    nf2_ms_b = nf2_magnetostatic_outputs['b'].to_value(u.G)

    metrics_magnetostatic = compute_metrics(nf2_ms_b, muram_B)
    metrics_force_free = compute_metrics(nf2_ff_b, muram_B)

    ########################################################
    # Plot metrics

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    ax = axs[0]
    ax.plot(metrics_magnetostatic['E_n'], np.arange(metrics_magnetostatic['E_n'].shape[0]) * Mm_per_pixel,
            label='NF2 magnetostatic', linestyle='dashed')
    ax.plot(metrics_force_free['E_n'], np.arange(metrics_force_free['E_n'].shape[0]) * Mm_per_pixel,
            label='NF2 force-free', linestyle='dotted')
    ax.set_xlabel(r"$E_\text{n}'$")

    ax = axs[1]
    ax.plot(metrics_magnetostatic['c_cs'], np.arange(metrics_magnetostatic['c_cs'].shape[0]) * Mm_per_pixel,
            label='NF2 magnetostatic', linestyle='dashed')
    ax.plot(metrics_force_free['c_cs'], np.arange(metrics_force_free['c_cs'].shape[0]) * Mm_per_pixel,
            label='NF2 force-free', linestyle='dotted')
    ax.set_xlabel(r'$c_\text{cs}$')

    ax = axs[2]
    ax.plot(metrics_magnetostatic['c_vec'], np.arange(metrics_magnetostatic['c_vec'].shape[0]) * Mm_per_pixel,
            label='NF2 magnetostatic', linestyle='dashed')
    ax.plot(metrics_force_free['c_vec'], np.arange(metrics_force_free['c_vec'].shape[0]) * Mm_per_pixel,
            label='NF2 force-free', linestyle='dotted')
    ax.set_xlabel(r'$c_\text{vec}$')

    ax = axs[3]
    ax.plot(metrics_magnetostatic['E_m'], np.arange(metrics_magnetostatic['E_m'].shape[0]) * Mm_per_pixel,
            label='NF2 magnetostatic', linestyle='dashed')
    ax.plot(metrics_force_free['E_m'], np.arange(metrics_force_free['E_m'].shape[0]) * Mm_per_pixel,
            label='NF2 force-free', linestyle='dotted')
    ax.set_xlabel(r"$E_\text{m}'$")

    ax = axs[4]
    ax.plot(metrics_magnetostatic['eps'], np.arange(metrics_magnetostatic['eps'].shape[0]) * Mm_per_pixel,
            label='NF2 magnetostatic', linestyle='dashed')
    ax.plot(metrics_force_free['eps'], np.arange(metrics_force_free['eps'].shape[0]) * Mm_per_pixel,
            label='NF2 force-free', linestyle='dotted')
    ax.set_xlabel(r'$\epsilon$')

    axs[0].set_ylabel('Height [Mm]')
    axs[-1].legend()

    [ax.set_xlim(0, 1) for ax in axs]
    [ax.set_ylim(0, 100) for ax in axs]

    axs[-1].set_xlim(0.7, 1.3)
    axs[-1].axvline(1, color='black', linestyle='dotted')

    plt.savefig(os.path.join(args.out_path, 'metrics.png'))
    plt.close(fig)
