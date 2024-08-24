import argparse
import os
from urllib.request import urlretrieve

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pfsspy
import torch
from astropy import constants
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
from pfsspy import tracing
from sunpy.map import Map, all_coordinates_from_map, make_heliographic_header
from sunpy.visualization.colormaps import cm

from nf2.evaluation.output import SphericalOutput

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files')
parser.add_argument('--cr', type=int, help='Carrington rotation number')

args = parser.parse_args()
nf2_path = args.nf2_path
out_path = args.out_path

os.makedirs(out_path, exist_ok=True)

# download AIA synoptic map from NJIT server
aia_synoptic_path = os.path.join(out_path, 'aia_synoptic.fits')
if not os.path.exists(aia_synoptic_path):
    urlretrieve(f'https://sun.njit.edu/coronal_holes/data/aia193_synmap_cr{args.cr}.fits', aia_synoptic_path)


def _field_line_endpoint(field_lines):
    field_lines = np.array([np.abs(fline._r[0] - fline._r[-1]) if len(fline._r) > 0 else 0 for fline in field_lines])
    return field_lines + 1



mag_r_map = Map("/glade/work/rjarolim/data/global/fd_2173/hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits")#f'/glade/work/rjarolim/data/global/synoptic/hmi.synoptic_mr_polfil_720s.{args.cr}.Mr_polfil.fits')
mag_r_map = mag_r_map.resample([180, 90] * u.pix)
nrho = 100
rss = 1.2
pfss_in = pfsspy.Input(mag_r_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

r = constants.R_sun
coords = all_coordinates_from_map(mag_r_map)
seeds = coords.ravel()
seeds = SkyCoord(lat=seeds.lat, lon=seeds.lon, radius=r, frame=coords.frame)

# NF2 field solution
print('Tracing field lines from NF2...')
nf2_out = SphericalOutput(nf2_path)
field_lines = nf2_out.trace(seeds)
field_map = np.array([np.abs(np.linalg.norm(fline[0]) - np.linalg.norm(fline[-1])) for fline in field_lines]).reshape(coords.shape)
field_map += 1

# potential field solution
print('Tracing field lines from PFSS...')
tracer = tracing.FortranTracer(max_steps=5000)
field_lines = tracer.trace(seeds, pfss_out)
# potential_field_map = field_lines.polarities.reshape(coords.shape)
potential_field_map = _field_line_endpoint(field_lines).reshape(coords.shape)



# load AIA synoptic map
aia_map = Map(aia_synoptic_path)
shape = aia_map.data.shape
carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape,
                                       frame='carrington', map_center_longitude=180 * u.deg)
aia_map = Map(aia_map.data, carr_header)
aia_map = aia_map.reproject_to(mag_r_map.wcs)

# plot field map with projection
fig, axs = plt.subplots(5, 1, figsize=(12, 24), subplot_kw={'projection': aia_map})

im = axs[0].imshow(aia_map.data, origin='lower', cmap=cm.sdoaia193)
plt.colorbar(im, label='Intensity [DN/s]')

cmap = plt.get_cmap('gray')
cmap.set_bad('yellow')
im = axs[1].imshow(aia_map.data, origin='lower', cmap=cm.sdoaia193)
plt.colorbar(im, label='Intensity [DN/s]')
axs[1].contourf(field_map, levels=[1.05, 1.5], colors=['blue'], origin='lower', alpha=0.3)
axs[1].contourf(potential_field_map, levels=[1.05, 1.5], colors=['red'], origin='lower', alpha=0.3)

im = axs[2].imshow(field_map, cmap=cmap, origin='lower', vmin=1, vmax=rss)
plt.colorbar(im, label='Endpoint [$R_{\odot}$]')

im = axs[3].imshow(potential_field_map, cmap=cmap, origin='lower', vmin=1, vmax=rss)
plt.colorbar(im, label='Endpoint [$R_{\odot}$]')

im = axs[4].imshow(mag_r_map.data, origin='lower', cmap=cmap, vmin=-500, vmax=500)
plt.colorbar(im, label='Magnetic Field Strength [G]')

# outline bright and dark regions
# smoothed_aia = block_reduce(aia_map.data, (4, 4), np.nanmean)
# levels = [np.nanpercentile(smoothed_aia, 10), np.nanpercentile(smoothed_aia, 80)]
# axs[0].contour(smoothed_aia, levels=levels,
#                colors=['blue', 'red'], extent=extent, linewidths=3, origin='lower')
# axs[1].contour(smoothed_aia, levels=levels,
#                colors=['blue', 'red'], extent=extent, linewidths=3, origin='lower')
# axs[2].contour(smoothed_aia, levels=levels,
#                colors=['blue', 'red'], extent=extent, linewidths=3, origin='lower')
# axs[3].contour(smoothed_aia, levels=levels,
#                colors=['blue', 'red'], extent=extent, linewidths=3, origin='lower')

# axis labels
[ax.set_xlabel('Carrington Longitude [deg]') for ax in axs]
[ax.set_ylabel('Carrington Latitude [deg]') for ax in axs]
axs[0].set_title('AIA 193')
axs[1].set_title('Overlay')
axs[2].set_title('NF2')
axs[3].set_title('PFSS')
axs[4].set_title('HMI')

plt.tight_layout()
plt.savefig(os.path.join(out_path, f'open_fields.jpg'))
plt.close()
