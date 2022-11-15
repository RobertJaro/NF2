import argparse
import glob
import os
import tarfile
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from tqdm import tqdm

from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.unpack import load_cube

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()

series_base_path = args.nf2_path
evaluation_path = os.path.join(series_base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

nf2_paths = sorted(glob.glob(os.path.join(series_base_path, '**', 'extrapolation_result.nf2')))
Mm_per_pix = 360e-3
cm_per_pix = (Mm_per_pix * args.strides * 1e8)
z_pixels = int(np.ceil(20 / (Mm_per_pix)))  # 20 Mm --> pixels; bin1

# save results as npy files
free_energy_files = []
for path in tqdm(nf2_paths):
    save_path = os.path.join(evaluation_path, '%s.npy' % path.split('/')[-2])
    if os.path.exists(save_path):
        free_energy_files += [save_path]
        continue
    b = load_cube(path, progress=False, z=z_pixels, strides=args.strides)
    free_me = get_free_mag_energy(b, progress=False)
    np.save(save_path, free_me)
    free_energy_files += [save_path]

series_dates = [datetime.strptime(os.path.basename(f)[:14], '%Y%m%d_%H%M%S') for f in free_energy_files]

free_energy_series = []
# plot the energy depletion
me_history = [None] * 4
for f, d in zip(free_energy_files, series_dates):
    free_me = np.load(f)
    prev_me = me_history.pop(0)
    if prev_me is None:
        prev_me = free_me
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(free_me.sum(2).transpose(), vmin=0, vmax=1e5, origin='lower', cmap='jet', )
    axs[0].set_title(d.isoformat(' '))
    axs[0].set_axis_off()
    axs[1].imshow(np.clip(free_me - prev_me, 0, None).sum(2).transpose(), vmin=0, vmax=1e4, origin='lower',
                  cmap='jet', )
    axs[1].set_title(d.isoformat(' '))
    axs[1].set_axis_off()
    me_history += [free_me]
    free_energy_series += [free_me.sum() * cm_per_pix ** 3]
    #
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_path, 'free_energy_%s.jpg' % d.isoformat('T')))
    plt.close()

flare_list = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/goes_flares_integrated.csv',
                         parse_dates=['end_time', 'event_date', 'peak_time', 'start_time'])
flare_list = flare_list[(flare_list.peak_time >= min(series_dates)) & (flare_list.peak_time <= max(series_dates))]

x_dates = date2num(series_dates)
date_format = DateFormatter('%d-%H:%M')

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.xaxis_date()
ax.set_xlim(x_dates[0], x_dates[-1])
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

ax.plot(x_dates, np.array(free_energy_series) * 1e-33)
ax.set_ylabel('Energy\n[$10^{33}$ erg]')

goes_mapping = {c: 10 ** (i) for i, c in enumerate(['B', 'C', 'M', 'X'])}
for t, goes_class in zip(flare_list.peak_time, flare_list.goes_class):
    flare_intensity = np.log10(float(goes_class[1:]) * goes_mapping[goes_class[0]])
    if flare_intensity >= np.log10(1 * 10 ** 2):
        ax.axvline(x=date2num(t), linestyle='dotted', c='black')
    if flare_intensity >= np.log10(1 * 10 ** 3):
        ax.axvline(x=date2num(t), linestyle='dotted', c='red')

plt.tight_layout()
plt.savefig(os.path.join(evaluation_path, 'free_energy_series.jpg'))
plt.close()

tar = tarfile.open(os.path.join(evaluation_path, 'free_energy_maps.tar.gz'), "w:gz")
for name in glob.glob(os.path.join(evaluation_path, 'free_energy_*.jpg'), recursive=True):
    tar.add(name, arcname=os.path.basename(name))

tar.close()
