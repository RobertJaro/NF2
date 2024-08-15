import argparse
import glob
import os
from copy import copy
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.sharp.convert_series import load_results


def _plot_map_videos(times, maps, wcs, euv_path, result_path, Mm_per_pixel):
    euv_files = np.array(sorted(glob.glob(euv_path)))
    euv_dates = np.array([parse(fits.getheader(f, 1)['DATE-OBS']) for f in euv_files])

    os.makedirs(result_path, exist_ok=True)

    j_norm = LogNorm()
    free_energy_norm = LogNorm(1e9, 1e14)

    for i, time in tqdm(enumerate(times), total=len(times)):
        euv_file = euv_files[np.argmin(np.abs(euv_dates - time))]
        euv_map = Map(euv_file)
        exposure = euv_map.exposure_time.to_value(u.s)
        euv_map = euv_map.reproject_to(wcs[i][0])

        b_0 = maps['b_0'][i].value
        current_density_map = maps['current_density_map'][i].to_value(u.G * u.cm * u.s ** -1)
        free_energy_map = maps['free_energy_map'][i].to_value(u.erg * u.cm ** -2)

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))

        ax = axs[0, 0]

        extent = np.array([0, b_0.shape[0], 0, b_0.shape[1]]) * Mm_per_pixel
        im = ax.imshow(b_0.T, origin='lower', cmap='gray', extent=extent, norm=Normalize(vmin=-2000, vmax=2000))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Magnetic Field - B$_z$')

        ax = axs[0, 1]
        cm = copy(plt.get_cmap('sdoaia131'))
        cm.set_bad('black')
        im = ax.imshow(euv_map.data / exposure, origin='lower', cmap=cm, extent=extent, norm=LogNorm(vmin=5, vmax=1e3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[DN / s]')
        ax.set_title('EUV - AIA 131')

        ax = axs[1, 0]

        im = ax.imshow(current_density_map.T, origin='lower', cmap='inferno', extent=extent, norm=j_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[G cm / s]')
        ax.set_title('Current Density - |J|')

        ax = axs[1, 1]
        free_energy_map[free_energy_map < 1e9] = 1e9
        im = ax.imshow(free_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=free_energy_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[erg / cm$^2$]')
        ax.set_title('Free Magnetic Energy')

        [ax.set_xlabel('X [Mm]') for ax in axs[1, :]]
        [ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]

        fig.suptitle(f'Time: {time.isoformat(" ", timespec="minutes")}')

        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f'{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--euv_path', type=str, help='path to the EUV files.', required=False, default=None)
    args = parser.parse_args()

    output = load_results(args.pkl_path)

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    times = output['times']
    wcs = output['wcs']
    maps = output['maps']
    Mm_per_pixel = output['Mm_per_pixel']

    _plot_map_videos(times, maps, wcs, args.euv_path, result_path, Mm_per_pixel)

if __name__ == '__main__':
    main()