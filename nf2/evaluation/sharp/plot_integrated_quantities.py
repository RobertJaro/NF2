import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter

from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.net import Fido
from sunpy.net import attrs as a

from nf2.evaluation.sharp.convert_series import load_results

from sunpy import timeseries as ts

def main():
    pass


def _plot_integrated_qunatities(times, integrated_quantities, height_distribution, result_path, Mm_per_pixel):
    pass


# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
args = parser.parse_args()

output = load_results(args.pkl_path)

# plot integrated quantities
# _plot_integrated_qunatities(output['times'], output['integrated_quantities'], output['height_distribution'],
#                             args.result_path, output['Mm_per_pixel'])
times = output['times']
integrated_quantities = output['integrated_quantities']
height_distribution = output['height_distribution']
result_path = args.result_path
Mm_per_pixel = output['Mm_per_pixel']

free_energy = integrated_quantities['free_energy'].to_value(u.erg) * 1e-32
#
free_energy_distribution = height_distribution['height_free_energy'].to_value(u.erg * u.cm ** -1)
df = pd.DataFrame(index=times, data={'idx': np.arange(len(times), dtype=int)})
df = df.groupby(pd.Grouper(freq='12Min')).first()

filled_energy_distribution = np.ones((len(df), *free_energy_distribution.shape[1:])) * np.nan
cond = ~np.isnan(df['idx'].values)
filled_energy_distribution[cond, :] = free_energy_distribution
filled_times = df.index.to_pydatetime()

result_goes = Fido.search(a.Time(times[0], times[-1]),
                          a.Instrument.xrs & a.goes.SatelliteNumber(16) & a.Resolution("avg1m"))
file_goes = Fido.fetch(result_goes)
goes = ts.TimeSeries(file_goes, source='XRS', concatenate=True)

date_format = DateFormatter('%d-%H:%M')


fig, full_axs = plt.subplots(3, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [1, 0.03]})
axs = full_axs[:, 0]
[ax.axis('off') for ax in full_axs[:, 1]]
# make date axis
for ax in axs:
    ax.xaxis_date()
    ax.set_xlim(times[0], times[-1])
    ax.xaxis.set_major_formatter(date_format)

fig.autofmt_xdate()

ax = axs[0]
ax.plot(times, free_energy)
ax.set_ylabel('Free Energy\n[$10^{32}$ erg]')

ax = axs[1]
dt = np.diff(filled_times)[0] / 2
dz = Mm_per_pixel
max_height = filled_energy_distribution.shape[1] * Mm_per_pixel
im = ax.imshow(filled_energy_distribution.T,  # average
                extent=(filled_times[0] - dt, filled_times[-1] + dt, -dz, max_height + dz), aspect='auto',
                origin='lower',
                cmap=cm.get_cmap('jet'), )
ax.set_ylabel('Height\n[Mm]')
ax.set_ylim(0, 20)
# add colorbar
full_axs[1, 1].axis('on')
cbar = fig.colorbar(im, cax=full_axs[1, 1], label='Free Energy Density\n' + r'[erg $cm^{-1}]$')
# cbar.formatter.set_powerlimits((0, 0))
# cbar.set_ticks([2e2, 4e2, 6e2, 8e2])

ax = axs[2]
ax.plot(goes.data.index, goes.data['xrsb'], label='1-8 $\AA$')
ax.legend( loc='upper left')
ax.set_yscale("log")
ax.set_ylim(1e-6, 1e-3)
ax.set_ylabel('GOES\n' + r'[W m$^{-2}$]')

for value in [1e-5, 1e-4]:
    ax.axhline(value, c='gray')


for value, label in zip([1e-6, 1e-5, 1e-4], ['C', 'M', 'X']):
    ax.text(1.02, value, label, transform=ax.get_yaxis_transform(), horizontalalignment='center')


for ax in axs:
    ax.axvline(x=datetime(2024, 5, 7, 0, 0), linestyle='dotted', c='black')
    ax.axvline(x=datetime(2024, 5, 7, 20, 18), linestyle='dotted', c='black')
    ax.axvline(x=datetime(2024, 5, 8, 4, 37), linestyle='dotted', c='black')
    ax.axvline(x=datetime(2024, 5, 9, 17, 23), linestyle='dotted', c='black')
    ax.axvline(x=datetime(2024, 5, 10, 6, 27), linestyle='dotted', c='black')
    # plot 60 degree line
    ax.axvline(x=datetime(2024, 5, 10, 15, 34), linestyle='solid', c='magenta')

# labels = ['A', 'B', 'C', 'M', 'X']
# centers = np.logspace(-7.5, -3.5, len(labels))
# for value, label in zip(centers, labels):
#     ax.text(1.02, value, label, transform=ax.get_yaxis_transform(), horizontalalignment='center')

# f_axs = axs[[0, 1, 3, 4]]
# if add_flares:
#     flares = Fido.search(a.Time(min(times), max(times)),
#                          a.hek.EventType("FL"),
#                          a.hek.OBS.Observatory == "GOES")["hek"]
#     goes_mapping = {c: 10 ** (i) for i, c in enumerate(['B', 'C', 'M', 'X'])}
#     for t, goes_class in zip(flares['event_peaktime'], flares['fl_goescls']):
#         flare_intensity = np.log10(float(goes_class[1:]) * goes_mapping[goes_class[0]])
#         if flare_intensity >= np.log10(5 * 10 ** 2):
#             [ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='black') for ax in f_axs]
#         if flare_intensity >= np.log10(1 * 10 ** 3):
#             [ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='red') for ax in f_axs]
plt.tight_layout()
plt.savefig(os.path.join(result_path, 'integrated_quantities.jpg'), dpi=300)
plt.close()
