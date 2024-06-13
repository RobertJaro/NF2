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
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.sharp.convert_series import load_results


def main():
    pass


def _plot_integrated_qunatities(times, integrated_quantities, height_distribution, result_path, Mm_per_pixel):
    pass


# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
parser.add_argument('--euv_path', type=str, help='path to the EUV files.', required=False, default=None)
args = parser.parse_args()

output = load_results(args.pkl_path)

euv_files = np.array(sorted(glob.glob(args.euv_path)))
euv_dates = np.array([parse(fits.getheader(f, 1)['DATE-OBS']) for f in euv_files])

result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

times = output['times']
wcs = output['wcs']
maps = output['maps']
Mm_per_pixel = output['Mm_per_pixel']

j_norm = LogNorm()
free_energy_norm = LogNorm(1e9, 1e14)

selected_times = [
    datetime(2024, 5, 8, 11, 0),
]

cond = [np.argmin(np.abs(times - t)) for t in selected_times]
times = times[cond]
wcs = wcs[cond]
maps = {k: v[cond] for k, v in maps.items()}


for i, time in tqdm(enumerate(times), total=len(times)):
    euv_file = euv_files[np.argmin(np.abs(euv_dates - time))]
    euv_map = Map(euv_file)
    exposure = euv_map.exposure_time.to_value(u.s)
    euv_map = euv_map.reproject_to(wcs[i][0])

    b_0 = maps['b_0'][i].value
    current_density_map = maps['current_density_map'][i].to_value(u.G * u.cm * u.s ** -1)
    free_energy_map = maps['free_energy_map'][i].to_value(u.erg * u.cm ** -2)

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    extent = np.array([0, b_0.shape[0], 0, b_0.shape[1]]) * Mm_per_pixel
    b0_im = ax.imshow(b_0.T, origin='lower', cmap='gray', extent=extent, norm=Normalize(vmin=-2000, vmax=2000))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # fig.colorbar(im, cax=cax, label='[G]')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(result_path, f'b0_{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    cm = copy(plt.get_cmap('sdoaia131'))
    cm.set_bad('black')
    euv_im = ax.imshow(euv_map.data / exposure, origin='lower', cmap=cm, extent=extent, norm=LogNorm(vmin=5, vmax=1e3))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # fig.colorbar(im, cax=cax, label='[DN / s]')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(result_path, f'euv_{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    cd_im = ax.imshow(current_density_map.T, origin='lower', cmap='inferno', extent=extent, norm=j_norm)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # fig.colorbar(im, cax=cax, label='[G cm / s]')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(result_path, f'j_{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    fe_im = ax.imshow(free_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=free_energy_norm)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # fig.colorbar(im, cax=cax, label='[erg / cm$^2$]')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(result_path, f'free_energy_{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
    plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
ax.set_axis_off()
cbar = fig.colorbar(b0_im, ax=ax, label='[G]')
# set ticks
cbar.set_ticks([-2000, -1000, 0, 1000, 2000])
plt.tight_layout(pad=0)
plt.savefig(os.path.join(result_path, f'b0_colorbar.jpg'), dpi=300)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
ax.set_axis_off()
fig.colorbar(euv_im, ax=ax, label='[DN / s]')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(result_path, f'euv_colorbar.jpg'), dpi=300)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
ax.set_axis_off()
fig.colorbar(cd_im, ax=ax, label='[G cm / s]')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(result_path, f'j_colorbar.jpg'), dpi=300)
plt.close()
