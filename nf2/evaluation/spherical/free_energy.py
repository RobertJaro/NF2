import os

import numpy as np
import pfsspy
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.util import cartesian_to_spherical, vector_cartesian_to_spherical
from nf2.evaluation.output import SphericalOutput

synoptic_file = '/glade/work/rjarolim/data/global/fd_2283/hmi.synoptic_mr_polfil_720s.2282.Mr_polfil.fits'
full_disc_file = '/glade/work/rjarolim/data/global/fd_2283/full_disk/hmi.b_720s.20240423_021200_TAI.Br.fits'
nf2_file = '/glade/work/rjarolim/nf2/spherical/2283_v01/extrapolation_result.nf2'

results_path = '/glade/work/rjarolim/nf2/spherical/2283_v01/results'
os.makedirs(results_path, exist_ok=True)

synoptic_map = Map(synoptic_file).resample([360, 180] * u.pix)
full_disc_map = Map(full_disc_file)

potential_r_map = full_disc_map.reproject_to(synoptic_map.wcs)
mask = np.isnan(potential_r_map.data)
potential_r_map.data[mask] = synoptic_map.data[mask]

model = SphericalOutput(nf2_file)

latitude_range = 0.398 - np.pi/2, 2.911 - np.pi/2
longitude_range = 1.673, 4.501

coords = np.stack(np.meshgrid(
np.linspace(1, 1.3, 128),
    np.linspace(*latitude_range, np.rad2deg(latitude_range[1] - latitude_range[0]).astype(int) * 2),
    np.linspace(*longitude_range, np.rad2deg(longitude_range[1] - longitude_range[0]).astype(int) * 2), indexing='ij'),
    -1)

# PFSS extrapolation
spherical_boundary_coords = SkyCoord(lon=coords[..., 2] * u.rad, lat=(coords[..., 1]) * u.rad,
                                     radius=coords[..., 0] * u.solRad, frame=potential_r_map.coordinate_frame)

model_out = model.load_spherical_coords(spherical_boundary_coords)
b = model_out['b'].to_value(u.G)
b = vector_cartesian_to_spherical(b, model_out['spherical_coords'])
j = model_out['j'].to_value(u.G / u.Mm)
j = vector_cartesian_to_spherical(j, model_out['spherical_coords'])

# potential_r_map.data[np.isnan(potential_r_map.data)] = 0

pfss_in = pfsspy.Input(potential_r_map, 128, 2.5)
pfss_out = pfsspy.pfss(pfss_in)

potential_shape = spherical_boundary_coords.shape  # required workaround for pfsspy spherical reshape
spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
spherical_boundary_values[..., 1] *= -1  # flip B_theta
potential_b = np.stack([spherical_boundary_values[..., 0],
                        spherical_boundary_values[..., 1],
                        spherical_boundary_values[..., 2]], -1)

# compute free energy
free_energy = (b ** 2).sum(-1) - (potential_b ** 2).sum(-1)

extent = None#[0, 360, -90, 90]
# plot free energy
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(free_energy.sum(0), origin='lower',
               norm=LogNorm(vmin=1, vmax=1e7),
               cmap='cividis', extent=extent)
ax.set_title('Free Energy')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
# grid with steps of 20
ax.grid()
# ax.set_xticks(np.arange(0, 360, 20))
# ax.set_yticks(np.arange(-80, 90, 20))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(os.path.join(results_path,
                         f'free_energy.jpg'),
            dpi=300)
plt.close(fig)

# free energy v2
free_energy = np.linalg.norm(b - potential_b, axis=-1)

extent = None#[0, 360, -90, 90]
# plot free energy
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(free_energy.sum(0), origin='lower',
               norm=LogNorm(vmin=10, vmax=1e4),
               cmap='cividis', extent=extent)
ax.set_title('Free Energy')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
# grid with steps of 20
ax.grid()
# ax.set_xticks(np.arange(0, 360, 20))
# ax.set_yticks(np.arange(-80, 80, 20))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(os.path.join(results_path, f'free_energy_v2.jpg'),
            dpi=300)
plt.close(fig)

# plot integrated currents
fig, axs = plt.subplots(1, 4, figsize=(15, 5)) # horizontal


ax = axs[0]
im = ax.imshow(b[0, :, :, 0], origin='lower', vmin=-500, vmax=500, cmap='gray', extent=extent)
ax.set_title('Photospheric Magnetic Field')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[1]
current_density = (j[..., :] ** 2).sum(-1) ** 0.5
j_norm = LogNorm(vmin=1, vmax=1e3)
im = ax.imshow(current_density.sum(0), origin='lower',
               norm=j_norm,
               cmap='plasma', extent=extent)
ax.set_title('Current Density')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[2]
current_density = (j[..., 1:] ** 2).sum(-1) ** 0.5
im = ax.imshow(current_density.sum(0), origin='lower',
               norm=j_norm,
               cmap='plasma', extent=extent)
ax.set_title('Horizontal Current Density')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax = axs[3]
current_density = (j[..., 0] ** 2) ** 0.5
im = ax.imshow(current_density.sum(0), origin='lower',
               norm=j_norm,
               cmap='plasma', extent=extent)
ax.set_title('Radial Current Density')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)

plt.savefig(os.path.join(results_path, f'currents.jpg'),
            dpi=300)
plt.close(fig)





# plot photospheric magnetic field
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(b[0, :, :, 0], origin='lower',
               vmin=-500, vmax=500,
               cmap='gray', extent=extent)
ax.set_title('Photospheric Magnetic Field')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
# ax.set_xticks(np.arange(0, 360, 20))
# ax.set_yticks(np.arange(-80, 80, 20))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(os.path.join(results_path, f'b.jpg'),
            dpi=300)
plt.close(fig)

# plot integrated magnetic field
fig, ax = plt.subplots(figsize=(10, 5))
energy = (b ** 2).sum(-1) ** 0.5
im = ax.imshow(energy.sum(0), origin='lower',
               norm=LogNorm(vmin=1e1, vmax=1e4),
               cmap='viridis', extent=extent)
ax.set_title('Integrated Magnetic Field')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.grid()
# ax.set_xticks(np.arange(0, 360, 20))
# ax.set_yticks(np.arange(-80, 80, 20))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(os.path.join(results_path, f'energy.jpg'),
            dpi=300)
plt.close(fig)
