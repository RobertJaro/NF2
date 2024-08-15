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
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.sharp.convert_series import load_results

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

times = output['times']
wcs = output['wcs']
maps = output['maps']
Mm_per_pixel = output['Mm_per_pixel']

j_norm = LogNorm()

selected_times = [
    datetime(2024, 5, 10, 5, 0, 0),
    datetime(2024, 5, 10, 6, 0, 0),
    datetime(2024, 5, 10, 7, 0, 0),
    datetime(2024, 5, 10, 8, 0, 0),
    datetime(2024, 5, 10, 9, 0, 0),
    datetime(2024, 5, 10, 10, 0, 0),
]

cm = copy(plt.get_cmap('sdoaia131'))
cm.set_bad('black')

fig, axs = plt.subplots(len(selected_times), 2, figsize=(5.6, 2 * len(selected_times)), sharex=True)

for i, time in tqdm(enumerate(selected_times), total=len(selected_times)):
    nf2_cond = np.argmin(np.abs(times - time))
    current_density_map = maps['current_density_map'][nf2_cond]
    current_density_map = current_density_map.to_value(u.G * u.cm * u.s ** -1)
    map_wcs = wcs[nf2_cond][0]
    #
    euv_file = euv_files[np.argmin(np.abs(euv_dates - time))]
    euv_map = Map(euv_file)
    exposure = euv_map.exposure_time.to_value(u.s)
    euv_map = euv_map.reproject_to(map_wcs)
    #
    extent = np.array([0, current_density_map.shape[0],
                       0, current_density_map.shape[1]]) * Mm_per_pixel
    #
    row = axs[i, :]
    #
    ax = row[0]
    cd_im = ax.imshow(current_density_map.T, origin='lower',
                      cmap='inferno', extent=extent, norm=j_norm)
    #
    ax = row[1]
    euv_im = ax.imshow(euv_map.data / exposure, origin='lower',
                       cmap=cm, extent=extent, norm=LogNorm(vmin=5, vmax=1e3))
    # add time as label on the right in bold
    ax.set_ylabel(time.isoformat(' ', timespec='minutes'), fontweight='bold')
    ax.yaxis.set_label_position("right")

divider = make_axes_locatable(axs[0, 0])
cax = divider.append_axes("top", size="5%", pad=0.05)
fig.colorbar(cd_im, cax=cax, label='Current Density [G cm / s]', orientation='horizontal')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

divider = make_axes_locatable(axs[0, 1])
cax = divider.append_axes("top", size="5%", pad=0.05)
fig.colorbar(euv_im, cax=cax, label='SDO/AIA 131 $\AA$ [DN / s]', orientation='horizontal')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')

[ax.set_xlim(100, 300) for ax in np.ravel(axs)]
[ax.set_ylim(20, 150) for ax in np.ravel(axs)]

[plt.setp(ax.get_xticklabels(), visible=False) for ax in np.ravel(axs[:-1, :])]
[plt.setp(ax.get_yticklabels(), visible=False) for ax in np.ravel(axs[:, 1:])]

[ax.set_xlabel('X [Mm]') for ax in np.ravel(axs[-1, :])]
[ax.set_ylabel('Y [Mm]') for ax in np.ravel(axs[:, 0])]

fig.tight_layout(pad=0.1)
fig.savefig(result_path, dpi=300)
plt.close()
