import glob
import os

import numpy as np
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import energy
from nf2.evaluation.output import SphericalOutput

file_1 = f'/glade/work/rjarolim/nf2/spherical/2173/20160205_200000_v02/extrapolation_result.nf2'
file_2 = f'/glade/work/rjarolim/nf2/spherical/2173/20160205_220000_v02/extrapolation_result.nf2'
results_path = '/glade/work/rjarolim/nf2/spherical/2173/results'

euv_files = sorted(glob.glob("/glade/work/rjarolim/data/global/fd_2173/euv/*.193.image_lev1.fits"))
euv_dates = np.array([parse(os.path.basename(f).split('.')[2]) for f in euv_files])

os.makedirs(results_path, exist_ok=True)

sampling = [128, 128, 128]
longitude_range = [150, 210]
latitude_range = [60, 120]
radius_range = [1.0, 1.3]
dr = ((radius_range[1] - radius_range[0]) * u.solRad).to_value(u.cm) / sampling[0]

alpha_norm = Normalize(vmin=0, vmax=10)
j_norm = LogNorm(vmin=1e10, vmax=3e12)
energy_norm = LogNorm(vmin=1e10, vmax=5e13)
euv_norm = LogNorm(vmin=50, vmax=5e3)

model_1 = SphericalOutput(file_1)
model_out_1 = model_1.load_spherical(radius_range * u.solRad,
                                     longitude_range=longitude_range * u.deg, latitude_range=latitude_range * u.deg,
                                     metrics=['j', 'alpha'], sampling=sampling, progress=True)

model_2 = SphericalOutput(file_2)
model_out_2 = model_2.load_spherical(radius_range * u.solRad,
                                     longitude_range=longitude_range * u.deg, latitude_range=latitude_range * u.deg,
                                     metrics=['j', 'alpha'], sampling=sampling, progress=True)

energy_1 = energy(model_out_1['b'].to_value(u.G))
energy_2 = energy(model_out_2['b'].to_value(u.G))

# energy_diff = -np.clip(energy_1 - energy_2, a_min=None, a_max=0)
# energy_diff = energy_2 - energy_1
energy_diff = np.clip(energy_2 - energy_1, a_min=0, a_max=None)

energy_map = energy_diff.sum(0) * dr

extent = [*longitude_range, *reversed(latitude_range)]

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(energy_map, origin='upper', cmap='Reds', extent=extent,
               # vmax=1e11, vmin=-1e11)
               norm=LogNorm(vmin=1e10, vmax=1e12))
               # norm=SymLogNorm(vmin=-1e12, vmax=1e12, linthresh=1e10))
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(os.path.join(results_path, 'energy_diff.jpg'), dpi=300)
plt.close(fig)
