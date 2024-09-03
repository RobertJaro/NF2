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

os.makedirs(result_path, exist_ok=True)

o_free_energy = integrated_quantities['free_energy'].to_value(u.erg) * 1e-32
o_energy = integrated_quantities['energy'].to_value(u.erg) * 1e-32
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

fig, axs = plt.subplots(4, 1, figsize=(12, 6))

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
ax.plot(times, energy)
ax.set_ylabel('Energy\n[$10^{32}$ erg]', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(energy) / dt * 1e3, color=color)
twin_ax.set_ylabel('$\Delta$ Energy\n [$10^{29}$ erg/s]', color=color)
twin_ax.spines['right'].set_color(color)
twin_ax.yaxis.label.set_color(color)
twin_ax.tick_params(axis='y', colors=color)

ax = axs[2]
color = 'tab:blue'
ax.plot(times, free_energy, color=color)
ax.set_ylabel('Free Energy\n[$10^{32}$ erg]', color=color)
ax.spines['left'].set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(axis='y', colors=color)

twin_ax = ax.twinx()
color = 'tab:red'
twin_ax.plot(times, np.gradient(free_energy) / dt * 1e3, color=color)
twin_ax.set_ylabel('$\Delta$ Free Energy\n [$10^{29}$ erg/s]', color=color)
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

m_flare_times = [
    (datetime(2024, 5, 7, 8, 18), datetime(2024, 5, 7, 8, 40)),  # M1.3
    (datetime(2024, 5, 7, 11, 40), datetime(2024, 5, 7, 12, 7)),  # M2.5
    (datetime(2024, 5, 7, 20, 18), datetime(2024, 5, 7, 20, 34)),  # M2.1
    (datetime(2024, 5, 7, 21, 42), datetime(2024, 5, 7, 22, 21)),  # M3.3
    #
    (datetime(2024, 5, 8, 2, 16), datetime(2024, 5, 8, 2, 36)),  # M3.4
    (datetime(2024, 5, 8, 3, 38), datetime(2024, 5, 8, 3, 46)),  # M2.0
    (datetime(2024, 5, 8, 3, 19), datetime(2024, 5, 8, 3, 46)),  # M1.9
    (datetime(2024, 5, 8, 4, 20), datetime(2024, 5, 8, 4, 49)),  # M3.6
    (datetime(2024, 5, 8, 6, 44), datetime(2024, 5, 8, 7, 10)),  # M7.2
    (datetime(2024, 5, 8, 11, 26), datetime(2024, 5, 8, 12, 22)),  # M8.7
    # (datetime(2024, 5, 8, 17, 32), datetime(2024, 5, 8, 18, 0)),  # M7.9
    # (datetime(2024, 5, 8, 19, 15), datetime(2024, 5, 8, 19, 29)),  # M2.0
    #
    (datetime(2024, 5, 9, 3, 7), datetime(2024, 5, 9, 3, 23)),  # M4.0
    (datetime(2024, 5, 9, 3, 23), datetime(2024, 5, 9, 3, 49)),  # M4.6
    (datetime(2024, 5, 9, 4, 44), datetime(2024, 5, 9, 4, 55)),  # M1.7
    (datetime(2024, 5, 9, 6, 3), datetime(2024, 5, 9, 6, 31)),  # M2.3
    (datetime(2024, 5, 9, 6, 24), datetime(2024, 5, 9, 6, 31)),  # M2.5
    (datetime(2024, 5, 9, 11, 52), datetime(2024, 5, 9, 12, 2)),  # M3.1
    (datetime(2024, 5, 9, 12, 5), datetime(2024, 5, 9, 12, 20)),  # M2.9
    (datetime(2024, 5, 9, 13, 16), datetime(2024, 5, 9, 13, 29)),  # M3.7
    # (datetime(2024, 5, 8, 22, 4), datetime(2024, 5, 8, 23, 30)),  # M9.9
    # (datetime(2024, 5, 8, 22, 5), datetime(2024, 5, 8, 22, 45)),  # M9.7
    (datetime(2024, 5, 9, 22, 24), datetime(2024, 5, 9, 22, 47)),  # M2.6
    (datetime(2024, 5, 9, 23, 44), datetime(2024, 5, 9, 23, 55)),  # M1.6
    #
    (datetime(2024, 5, 10, 0, 10), datetime(2024, 5, 10, 0, 22)),  # M1.5
    (datetime(2024, 5, 10, 6, 14), datetime(2024, 5, 10, 6, 27)),  # M1.4
    (datetime(2024, 5, 10, 10, 10), datetime(2024, 5, 10, 10, 19)),  # M2.2
    (datetime(2024, 5, 10, 13, 58), datetime(2024, 5, 10, 14, 28)),  # M6.0
    (datetime(2024, 5, 10, 18, 26), datetime(2024, 5, 10, 18, 38)),  # M1.1
    (datetime(2024, 5, 10, 18, 38), datetime(2024, 5, 10, 19, 7)),  # M1.8
    (datetime(2024, 5, 10, 18, 57), datetime(2024, 5, 10, 19, 10)),  # M2.0
    (datetime(2024, 5, 10, 19, 35), datetime(2024, 5, 10, 19, 56)),  # M1.1
    (datetime(2024, 5, 10, 19, 56), datetime(2024, 5, 10, 20, 19)),  # M1.9
    (datetime(2024, 5, 10, 20, 59), datetime(2024, 5, 10, 21, 12)),  # M3.8
    #
    (datetime(2024, 5, 11, 4, 28), datetime(2024, 5, 11, 4, 51)),  # M1.4
]

x_flares = [
    # (datetime(2024, 5, 8, 21, 8), datetime(2024, 5, 8, 21, 58)),  # X1.0
    (datetime(2024, 5, 8, 4, 37), datetime(2024, 5, 8, 5, 32)),  # X1.0
    (datetime(2024, 5, 9, 8, 45), datetime(2024, 5, 9, 9, 36)),  # X2.2
    (datetime(2024, 5, 9, 17, 23), datetime(2024, 5, 9, 18, 0)),  # X1.1
    (datetime(2024, 5, 10, 6, 27), datetime(2024, 5, 10, 7, 6)),  # X4.0
    (datetime(2024, 5, 11, 1, 10), datetime(2024, 5, 11, 1, 39)),  # X5.8
]

for start, end in m_flare_times:
    [ax.axvspan(start, end, color='blue', alpha=0.2) for ax in axs]

for start, end in x_flares:
    [ax.axvspan(start, end, color='orange', alpha=1) for ax in axs]

nans = np.isnan(unsigned_flux)
# pairs of ranges
ranges = np.where(np.diff(nans) != 0)[0].reshape(-1, 2)
for start, end in ranges:
    [ax.axvspan(times[start], times[end], color='gray', alpha=0.3) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(result_path, 'integrated_energy.jpg'), dpi=300)
plt.close()