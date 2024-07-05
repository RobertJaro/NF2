import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.sharp.convert_series import load_results

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
parser.add_argument('--sharp_path', type=str, help='path to the output directory', required=False, default=None)
args = parser.parse_args()

Br = sorted(glob.glob(os.path.join(args.sharp_path, '*Br.fits')))
Bt = sorted(glob.glob(os.path.join(args.sharp_path, '*Bt.fits')))
Bp = sorted(glob.glob(os.path.join(args.sharp_path, '*Bp.fits')))

o_unsigned_flux = []
for Br_file, Bt_file, Bp_file in tqdm(zip(Br, Bt, Bp), total=len(Br)):
    # vector_B = np.stack([Map(Br_file).data, Map(Bt_file).data, Map(Bp_file).data], axis=-1)
    # flux = np.linalg.norm(vector_B, axis=-1).sum()
    flux = np.abs(Map(Br_file).data).sum()
    o_unsigned_flux.append(flux)

output = load_results(args.pkl_path)

o_times = output['times']
integrated_quantities = output['integrated_quantities']
result_path = args.result_path
Mm_per_pixel = output['Mm_per_pixel']

o_free_energy = integrated_quantities['free_energy'].to_value(u.erg) * 1e-32
o_energy = integrated_quantities['energy'].value * 1e-32
o_current = output['maps']['current_density_map'].sum((1, 2)) * Mm_per_pixel ** 2

date_format = DateFormatter('%d-%H:%M')

# fill gaps with NaNs
df = pd.DataFrame({'time': o_times, 'unsigned_flux': o_unsigned_flux,
                   'free_energy': o_free_energy, 'energy': o_energy, 'current': o_current})
df = df.set_index('time')
df = df.resample('12min').mean()
df = df.rolling('1H').mean()

dt = 12 * 60  # 12 minutes in seconds

times = df.index
unsigned_flux = df['unsigned_flux'].values
free_energy = df['free_energy'].values
energy = df['energy'].values
current = df['current'].values


fig, axs = plt.subplots(4, 1, figsize=(8, 6))

# make date axis
for ax in axs:
    ax.xaxis_date()
    ax.set_xlim(datetime(2024, 5, 7), times[-1])
    ax.xaxis.set_major_formatter(date_format)


fig.autofmt_xdate()

ax = axs[0]
color = 'tab:blue'
ax.plot(times, unsigned_flux, color=color)
ax.set_ylabel('Unsigned Flux\n[G]', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(unsigned_flux) / dt, color=color)
twin_ax.set_ylabel('$\Delta$ Unsigned Flux\n [G/s]', color=color)
twin_ax.spines['right'].set_color(color)
twin_ax.yaxis.label.set_color(color)
twin_ax.tick_params(axis='y', colors=color)

ax = axs[1]
color = 'tab:blue'
ax.plot(times, free_energy, color=color)
ax.set_ylabel('Free Energy\n[$10^{32}$ erg]', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(free_energy) / dt, color=color)
twin_ax.set_ylabel('$\Delta$ Free Energy\n [$10^{32}$ erg/s]', color=color)
twin_ax.spines['right'].set_color(color)
twin_ax.yaxis.label.set_color(color)
twin_ax.tick_params(axis='y', colors=color)


ax = axs[2]
color = 'tab:blue'
ax.plot(times, energy)
ax.set_ylabel('Energy\n[$10^{32}$ erg]', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(energy) / dt, color=color)
twin_ax.set_ylabel('$\Delta$ Energy\n [$10^{32}$ erg/s]', color=color)
twin_ax.spines['right'].set_color(color)
twin_ax.yaxis.label.set_color(color)
twin_ax.tick_params(axis='y', colors=color)

energy_ratio = free_energy / energy
ax = axs[3]
color = 'tab:blue'
ax.plot(times, energy_ratio, color=color)
ax.set_ylabel('E$_{free}$ / E', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(energy_ratio) / dt, color=color)
twin_ax.set_ylabel('$\Delta$ E$_{free}$ / E [1 / s]', color=color)
twin_ax.spines['right'].set_color(color)
twin_ax.yaxis.label.set_color(color)
twin_ax.tick_params(axis='y', colors=color)

for ax in axs:
    ax.axvline(x=datetime(2024, 5, 7, 11, 50), linestyle='dotted', c='black') # M2.5
    ax.axvline(x=datetime(2024, 5, 7, 21, 59), linestyle='dotted', c='black')  # M3.3
    ax.axvline(x=datetime(2024, 5, 7, 20, 22), linestyle='dotted', c='black')  # M2.1
    #
    ax.axvline(x=datetime(2024, 5, 8, 2, 27), linestyle='dotted', c='black')  # M3.4
    ax.axvline(x=datetime(2024, 5, 8, 12, 4), linestyle='dotted', c='black')  # M8.7
    # ax.axvline(x=datetime(2024, 5, 8, 17, 53), linestyle='dotted', c='black')  # M7.9
    # ax.axvline(x=datetime(2024, 5, 8, 19, 21), linestyle='dotted', c='black')  # M2.0
    # ax.axvline(x=datetime(2024, 5, 8, 22, 27), linestyle='dotted', c='black')  # M9.9
    # ax.axvline(x=datetime(2024, 5, 8, 21, 40), linestyle='dotted', c='red')  # X1.0
    # ax.axvline(x=datetime(2024, 5, 8, 20, 34), linestyle='dotted', c='black')  # M1.8
    ax.axvline(x=datetime(2024, 5, 8, 6, 53), linestyle='dotted', c='black')  # M7.2
    ax.axvline(x=datetime(2024, 5, 8, 5, 9), linestyle='dotted', c='red')  # X1.0
    ax.axvline(x=datetime(2024, 5, 8, 3, 30), linestyle='dotted', c='black')  # M3.6
    ax.axvline(x=datetime(2024, 5, 8, 3, 42), linestyle='dotted', c='black')  # M2.0
    ax.axvline(x=datetime(2024, 5, 8, 3, 27), linestyle='dotted', c='black')  # M1.9
    #
    ax.axvline(x=datetime(2024, 5, 9, 20, 34), linestyle='dotted', c='black')  # M1.8
    ax.axvline(x=datetime(2024, 5, 9, 3, 17), linestyle='dotted', c='black')  # M4.0
    ax.axvline(x=datetime(2024, 5, 9, 3, 32), linestyle='dotted', c='black')  # M4.6
    ax.axvline(x=datetime(2024, 5, 9, 4, 49), linestyle='dotted', c='black')  # M1.7
    ax.axvline(x=datetime(2024, 5, 9, 6, 12), linestyle='dotted', c='black')  # M2.3
    ax.axvline(x=datetime(2024, 5, 9, 6, 27), linestyle='dotted', c='black')  # M2.5
    ax.axvline(x=datetime(2024, 5, 9, 11, 56), linestyle='dotted', c='black')  # M3.1
    ax.axvline(x=datetime(2024, 5, 9, 12, 12), linestyle='dotted', c='black')  # M2.9
    ax.axvline(x=datetime(2024, 5, 9, 13, 23), linestyle='dotted', c='black')  # M3.7
    ax.axvline(x=datetime(2024, 5, 9, 22, 41), linestyle='dotted', c='black')  # M2.6
    ax.axvline(x=datetime(2024, 5, 9, 17, 44), linestyle='dotted', c='red')  # X1.1
    ax.axvline(x=datetime(2024, 5, 9, 9, 13), linestyle='dotted', c='red')  # X2.3
    ax.axvline(x=datetime(2024, 5, 9, 21, 40), linestyle='dotted', c='red')  # X1.0
    ax.axvline(x=datetime(2024, 5, 9, 19, 21), linestyle='dotted', c='black')  # M2.0
    ax.axvline(x=datetime(2024, 5, 9, 22, 27), linestyle='dotted', c='black')  # M9.9
    ax.axvline(x=datetime(2024, 5, 9, 22, 27), linestyle='dotted', c='black')  # M9.7
    #
    ax.axvline(x=datetime(2024, 5, 10, 0, 18), linestyle='dotted', c='black') # M1.5
    ax.axvline(x=datetime(2024, 5, 10, 6, 24), linestyle='dotted', c='black')  # M1.4
    ax.axvline(x=datetime(2024, 5, 10, 10, 14), linestyle='dotted', c='black')  # M2.2
    ax.axvline(x=datetime(2024, 5, 10, 14, 11), linestyle='dotted', c='black')  # M6.0
    ax.axvline(x=datetime(2024, 5, 10, 18, 32), linestyle='dotted', c='black')  # M1.1
    ax.axvline(x=datetime(2024, 5, 10, 14, 11), linestyle='dotted', c='black')  # M6.0
    ax.axvline(x=datetime(2024, 5, 10, 18, 48), linestyle='dotted', c='black')  # M1.8
    ax.axvline(x=datetime(2024, 5, 10, 19, 5), linestyle='dotted', c='black')  # M2.0
    ax.axvline(x=datetime(2024, 5, 10, 19, 35), linestyle='dotted', c='black')  # M1.1
    ax.axvline(x=datetime(2024, 5, 10, 20, 3), linestyle='dotted', c='black')  # M1.9
    ax.axvline(x=datetime(2024, 5, 10, 21, 8), linestyle='dotted', c='black')  # M3.8
    ax.axvline(x=datetime(2024, 5, 10, 6, 54), linestyle='dotted', c='red')  # X4.0
    ax.axvline(x=datetime(2024, 5, 10, 23, 51), linestyle='dotted', c='black')  # M1.6
    #
    nans = np.isnan(unsigned_flux)
    # pairs of ranges
    ranges = np.where(np.diff(nans) != 0)[0].reshape(-1, 2)
    for start, end in ranges:
        ax.axvspan(times[start], times[end], color='orange', alpha=0.3)


plt.tight_layout()
plt.savefig(os.path.join(result_path, 'integrated_energy.jpg'), dpi=300)
plt.close()
