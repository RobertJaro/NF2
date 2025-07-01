import argparse
import glob
import os.path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from nf2.evaluation.output import CartesianOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_paths', type=str,
                        help='path to the directory of the NF2 files. Uses glob search pattern. (<<your path>>/**/extrapolation_result.nf2)',
                        required=True)
    parser.add_argument('--out_path', type=str, help='path to the output directory', required=True)
    parser.add_argument('--Mm_per_pixel', type=float, help='Mm per pixel', required=False, default=1.44)
    args = parser.parse_args()

    nf2_paths = args.nf2_paths
    out_path = args.out_path
    Mm_per_pixel = args.Mm_per_pixel

    os.makedirs(out_path, exist_ok=True)

    nf2_files = sorted(glob.glob(nf2_paths, recursive=True))

    b_ensemble = []
    j_ensemble = []
    for f in tqdm(nf2_files, desc='Loading Ensemble'):
        model = CartesianOutput(f)
        out = model.load_cube(Mm_per_pixel=Mm_per_pixel, metrics=['j'])
        b_ensemble.append(out['b'])
        j_ensemble.append(out['metrics']['j'])

    b_ensemble = np.stack(b_ensemble).to_value(u.G)
    j_ensemble = np.stack(j_ensemble).to_value(u.G / u.s)

    mean_b_vector = np.mean(b_ensemble, axis=0, keepdims=True)
    b_uncertainty = np.mean(np.linalg.norm(b_ensemble - mean_b_vector, axis=-1) ** 2, axis=0) ** 0.5

    mean_j_vector = np.mean(j_ensemble, axis=0, keepdims=True)
    j_uncertainty = np.mean(np.linalg.norm(j_ensemble - mean_j_vector, axis=-1) ** 2, axis=0) ** 0.5

    np.savez(os.path.join(out_path, 'uncertainty.npz'), b_uncertainty=b_uncertainty, j_uncertainty=j_uncertainty,
             Mm_per_pixel=Mm_per_pixel)

    extent = [0, b_uncertainty.shape[0] * Mm_per_pixel, 0, b_uncertainty.shape[1] * Mm_per_pixel]

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    ax = axs[0]
    im = ax.imshow(b_uncertainty.mean(2).T, cmap='viridis', origin='lower', extent=extent)
    ax.set_title(r'$\vec{B}$')
    ax.set_xlabel('X [Mm]')
    ax.set_ylabel('Y [Mm]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$\delta$B [G]')


    ax = axs[1]
    im = ax.imshow(j_uncertainty.mean(2).T, cmap='inferno', origin='lower', extent=extent)
    ax.set_title(r'$\vec{J}$')
    ax.set_xlabel('X [Mm]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$\delta$J [G/s]')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'uncertainty.png'), dpi=300, transparent=True)
    plt.close()


    labels = np.arange(0.05, 0.85, 0.05, dtype=float)
    norm = LogNorm()
    fig, axs = plt.subplots(4, 5, figsize=(10, 8),
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.03], 'wspace': 0.4, 'hspace': 0.4,})

    for i in range(16):
        ax = axs[:, :-1].flatten()[i]
        j_map = np.linalg.norm(j_ensemble[i], axis=-1).sum(axis=2) * Mm_per_pixel
        im = ax.imshow(j_map.T, cmap='inferno', origin='lower', extent=extent, norm=norm)
        ax.set_title(f'$\lambda_\\text{{ff}}$={labels[i]:.02f}')

    [ax.set_xlabel('X [Mm]') for ax in axs[-1, :]]
    [ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]

    for ax in axs[:, -1]:
        # add colorbar with anchor to the left
        fig.colorbar(im, cax=ax, orientation='vertical', label='J [G Mm s$^{-1}$]')

    plt.savefig(os.path.join(out_path, 'j_maps.png'), dpi=300, transparent=True)
    plt.close()