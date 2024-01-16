import argparse
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pfsspy
from astropy import constants
from astropy.coordinates import SkyCoord
from pfsspy import tracing
from sunpy.map import Map, all_coordinates_from_map

from nf2.evaluation.global_field.common import NF2Output

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files')
parser.add_argument('--cr', type=int, help='Carrington rotation number')

args = parser.parse_args()
nf2_path = args.nf2_path
out_path = args.out_path
max_radius = 1.3

os.makedirs(out_path, exist_ok=True)


def _field_line_endpoint(field_lines):
    field_lines = np.array([np.abs(fline._r[0] - fline._r[-1]) if len(fline._r) > 0 else 0 for fline in field_lines])
    return field_lines + 1


# potential field solution
mag_r_map = Map(f'/glade/work/rjarolim/data/global/synoptic/hmi.synoptic_mr_polfil_720s.{args.cr}.Mr_polfil.fits')
mag_r_map = mag_r_map.resample([360, 180] * u.pix)
nrho = 100
rss = 2.5
pfss_in = pfsspy.Input(mag_r_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

r = constants.R_sun

# select equally distributed seeds
coords = all_coordinates_from_map(mag_r_map).ravel()
values = np.abs(mag_r_map.data).ravel()
sort_idx = np.argsort(values)
X2 = values[sort_idx]

full_idx = np.linspace(0, 1, len(X2))
selected_idx = np.linspace(0, 1, 4096)
selected_idx = np.argmin(np.abs(full_idx[None, :] - selected_idx[:, None]), axis=1)

seeds = coords[sort_idx][selected_idx]
seeds = SkyCoord(lat=seeds.lat, lon=seeds.lon, radius=r, frame=seeds.frame)

tracer = tracing.FortranTracer(max_steps=1000)
potential_field_lines = tracer.trace(seeds, pfss_out)

nf2_out = NF2Output(nf2_path, pfss_in)
field_lines = tracer.trace(seeds, nf2_out)

# plot 3D field lines
fig = plt.figure(figsize=(20, 10))


# u, v = np.mgrid[0:2 * np.pi:radial_b_map.shape[1] * 1j, 0:np.pi:radial_b_map.shape[0] * 1j]
# x = 1 * np.cos(u) * np.sin(v)
# y = 1 * np.sin(u) * np.sin(v)
# z = 1 * np.cos(v)
# color_mapping = plt.get_cmap('gray')(Normalize(-500, 500)(radial_b_map))
# ax.plot_surface(x.T, y.T, z.T, facecolors=color_mapping, cstride=2, rstride=2, zorder=10)

ax = fig.add_subplot(121, projection='3d')
for field_line in potential_field_lines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    coords = field_line.coords
    coords.representation_type = 'cartesian'
    radius = np.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2) / constants.R_sun
    coords.x[radius > 1.3] = np.nan
    coords.x[coords.x > 0] = np.nan
    ax.plot(coords.x / constants.R_sun,
            coords.y / constants.R_sun,
            coords.z / constants.R_sun,
            color=color, linewidth=1)

ax.view_init(azim=0, elev=0)

ax = fig.add_subplot(122, projection='3d')
for field_line in field_lines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    coords = field_line.coords
    coords.representation_type = 'cartesian'
    radius = np.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2) / constants.R_sun
    coords.x[radius > 1.3] = np.nan
    coords.x[coords.x > 0] = np.nan
    ax.plot(coords.x / constants.R_sun,
            coords.y / constants.R_sun,
            coords.z / constants.R_sun,
            color=color, linewidth=1)

ax.view_init(azim=0, elev=0)

plt.savefig(os.path.join(out_path, f'field_lines.jpg'), dpi=300)
plt.close()
