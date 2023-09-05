import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

mag_reference = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/401/hmi.sharp_cea_720s.401.20110309_163600_TAI.Br.fits')
euv_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/aia.lev1_euv_12s.2011-03-09T163413Z.171.image_lev1.fits')

reprojected_euv_map = euv_map.reproject_to(mag_reference.wcs)

Mm_per_pixel = 0.36
extent = np.array([0, mag_reference.data.shape[1], 0, mag_reference.data.shape[0]]) * Mm_per_pixel

fig = plt.figure(figsize=(8, 4))

ax = plt.subplot(111)

ax.imshow(reprojected_euv_map.data, origin='lower', cmap=cm.sdoaia171, norm=ImageNormalize(vmin=0, stretch=AsinhStretch(0.01)), extent=extent)
ax.contour(mag_reference.data, origin='lower', colors=['blue', 'red'], levels=[-1000, 1000], extent=extent)

ax.set_xlabel('X [Mm]', size=20)
ax.set_ylabel('Y [Mm]', size=20)
ax.tick_params(labelsize=14)
ax.tick_params(axis='x', bottom=True, top=False)
ax.tick_params(axis='y', left=True, right=False)
ax.set_xlim(50, 220)
fig.tight_layout()
fig.savefig('/gpfs/gpfs0/robert.jarolim/multi_height/solis_401_chromospheric_pf_v3/evaluation/euv.png', dpi=300, transparent=True)
plt.close()