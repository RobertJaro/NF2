import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

mag_reference = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/401/hmi.sharp_cea_720s.401.20110309_163600_TAI.Br.fits')
euv_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/aia.lev1_euv_12s.2011-03-09T163413Z.171.image_lev1.fits')

reprojected_euv_map = euv_map.reproject_to(mag_reference.wcs)

fig = plt.figure(figsize=(8, 4))

ax = plt.subplot(111, projection=mag_reference)

ax.imshow(reprojected_euv_map.data, origin='lower', cmap=cm.sdoaia171, norm=ImageNormalize(vmin=0, stretch=AsinhStretch(0.01)))
ax.contour(mag_reference.data, origin='lower', colors=['black', 'white'], levels=[-1000, 1000])

ax.set_xlabel('Heliographic Longitutude', size=20)
ax.set_ylabel('Heliographic Latitude', size=20)
ax.tick_params(labelsize=14)
ax.tick_params(axis='x', bottom=True, top=False)
ax.tick_params(axis='y', left=True, right=False)
fig.tight_layout()
fig.savefig('/gpfs/gpfs0/robert.jarolim/multi_height/solis_401_chromospheric_pf_v3/evaluation/euv.png', dpi=300, transparent=True)
plt.close()