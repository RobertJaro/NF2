import glob
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.colors import SymLogNorm
from sunpy.coordinates import frames
from sunpy.map import Map, make_heliographic_header, all_coordinates_from_map

from nf2.data.util import vector_cartesian_to_spherical
from nf2.evaluation.output import SphericalOutput

import matplotlib.pyplot as plt

result_path = '/glade/work/rjarolim/nf2/spherical/2283_00_v01'
pre_eruption = SphericalOutput('/glade/work/rjarolim/nf2/spherical/2283_00_v01/extrapolation_result.nf2')
post_eruption = SphericalOutput('/glade/work/rjarolim/nf2/spherical/2283_04_v01/extrapolation_result.nf2')

latitude_range = np.deg2rad(-45), np.deg2rad(45)
longitude_range = 1.673, 4.501

coords = np.stack(np.meshgrid(
np.linspace(1.0, 1.3, 128),
    np.linspace(*latitude_range, np.rad2deg(latitude_range[1] - latitude_range[0]).astype(int) * 2),
    np.linspace(*longitude_range, np.rad2deg(longitude_range[1] - longitude_range[0]).astype(int) * 2), indexing='ij'),
    -1)

# PFSS extrapolation
spherical_boundary_coords = SkyCoord(lon=coords[..., 2] * u.rad, lat=(coords[..., 1]) * u.rad,
                                     radius=coords[..., 0] * u.solRad, frame=frames.HeliographicCarrington)

pre_eruption_out = pre_eruption.load_spherical_coords(spherical_boundary_coords, progress=True, batch_size=4096)
post_eruption_out = post_eruption.load_spherical_coords(spherical_boundary_coords, progress=True, batch_size=4096)

pre_j = pre_eruption_out['j'].to_value(u.G / u.Mm)
pre_j = vector_cartesian_to_spherical(pre_j, pre_eruption_out['spherical_coords'])[..., 1:]
post_j = post_eruption_out['j'].to_value(u.G / u.Mm)
post_j = vector_cartesian_to_spherical(post_j, post_eruption_out['spherical_coords'])[..., 1:]

pre_current_density = np.linalg.norm(pre_j, axis=-1)
post_current_density = np.linalg.norm(post_j, axis=-1)

current_change = post_current_density - pre_current_density
current_change = np.clip(current_change, a_min=None, a_max=0)

# translate AIA maps to spherical coordinates
shape = (720, 1440)
aia_maps = sorted(glob.glob('/glade/work/rjarolim/data/global/fd_2283/aia.lev1_euv_12s.*.131.image_lev1.fits'))

aia_data = []

for f in aia_maps:
    aia_map = Map(f)
    carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
    aia_map = aia_map.reproject_to(carr_header)
    d = aia_map.data
    d = np.roll(d, 720, axis=1)
    aia_data.append(d)

pre_aia = aia_data[0]
post_aia = aia_data[-1]
aia_integrated = np.mean(aia_data, axis=0)


# plot current change maps
extent = [np.rad2deg(longitude_range[0]), np.rad2deg(longitude_range[1]),
            np.rad2deg(latitude_range[0]), np.rad2deg(latitude_range[1])]

fig, axs = plt.subplots(2, 3, figsize=(15, 5))

ax = axs[0, 0]
im = ax.imshow(pre_aia, origin='lower', cmap='sdoaia131',
               extent=[0, 360, -90, 90], norm=SymLogNorm(linthresh=10, vmin=1, vmax=2e3))
plt.colorbar(im, ax=ax, label='Intensity [DN/s]')
ax.set_title('Pre-Eruption')
ax.set_xlim(*extent[:2])
ax.set_ylim(*extent[2:])

ax = axs[0, 1]
im = ax.imshow(post_aia, origin='lower', cmap='sdoaia131',
               extent=[0, 360, -90, 90], norm=SymLogNorm(linthresh=10, vmin=1, vmax=2e3))
plt.colorbar(im, ax=ax, label='Intensity [DN/s]')
ax.set_title('Post-Eruption')
ax.set_xlim(*extent[:2])
ax.set_ylim(*extent[2:])

ax = axs[0, 2]
im = ax.imshow(aia_integrated, origin='lower', cmap='RdBu_r',
               extent=[0, 360, -90, 90], norm=SymLogNorm(linthresh=10, vmin=-2e3, vmax=2e3))
plt.colorbar(im, ax=ax, label='Intensity Change [DN/s]')
ax.set_title('Integrated Intensity')
ax.set_xlim(*extent[:2])
ax.set_ylim(*extent[2:])

ax = axs[1, 0]
im = ax.imshow(pre_current_density.sum(0), origin='lower', cmap='inferno',
               norm=SymLogNorm(linthresh=10, vmin=0, vmax=1000), extent=extent)
plt.colorbar(im, ax=ax, label='Current Density [G/Mm]')
ax.set_title('Pre-Eruption')

ax = axs[1, 1]
im = ax.imshow(post_current_density.sum(0), origin='lower', cmap='inferno',
               norm=SymLogNorm(linthresh=10, vmin=0, vmax=1000), extent=extent)
plt.colorbar(im, ax=ax, label='Current Density [G/Mm]')
ax.set_title('Post-Eruption')

ax = axs[1, 2]
im = ax.imshow(current_change.sum(0), origin='lower',
               norm=SymLogNorm(linthresh=100, vmin=-1000, vmax=1000),
                cmap='RdBu_r', extent=extent)
plt.colorbar(im, ax=ax, label='Current Density Change [G/Mm]')

[ax.set_xlabel('Longitude [deg]') for ax in np.ravel(axs)]
axs[0, 0].set_ylabel('Latitude [deg]')
axs[1, 0].set_ylabel('Latitude [deg]')

fig.tight_layout()
plt.savefig(os.path.join(result_path, 'current_change.png'), dpi=300, transparent=True)
plt.close()