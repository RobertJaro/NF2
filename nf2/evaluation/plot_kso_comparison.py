import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from astropy import units as u

mag_reference = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/401/hmi.sharp_cea_720s.401.20110309_163600_TAI.Br.fits')
kso_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/kanz_halph_fi_20110309_151759.fts.gz')

angle = -kso_map.meta["angle"] * u.deg

my_coord = SkyCoord(11*u.arcsec, 0*u.arcsec, obstime=kso_map.date, observer = 'earth', frame=frames.Helioprojective)
my_header = make_fitswcs_header(kso_map.data, my_coord, scale=u.Quantity(kso_map.scale), rotation_angle=angle,
                                reference_pixel=u.Quantity(kso_map.reference_pixel), )
new_kso_map = Map(kso_map.data, my_header)


reprojected_kso_map = new_kso_map.reproject_to(mag_reference.wcs)

Mm_per_pixel = 0.36

extent = np.array([0, mag_reference.data.shape[1], 0, mag_reference.data.shape[0]]) * Mm_per_pixel

fig = plt.figure(figsize=(8, 4))

ax = plt.subplot(111)

ax.imshow(reprojected_kso_map.data, origin='lower', cmap='gray', extent=extent)
ax.contour(mag_reference.data, origin='lower', colors=['blue', 'red'], levels=[-1000, 1000], extent=extent)

ax.tick_params(labelsize=14)
ax.tick_params(axis='x', bottom=True, top=False)
ax.tick_params(axis='y', left=True, right=False)
ax.set_xlabel('X [Mm]', size=20)
# ax.set_ylabel('Y [Mm]', size=20)
ax.yaxis.set_ticklabels([])
ax.set_xlim(50, 220)
fig.tight_layout()
fig.savefig('/gpfs/gpfs0/robert.jarolim/multi_height/solis_401_chromospheric_pf_v3/evaluation/kso.png', dpi=300,
            transparent=True)
plt.close()
