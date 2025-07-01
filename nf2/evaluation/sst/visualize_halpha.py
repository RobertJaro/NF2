import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from astropy import units as u
from sunpy.visualization.colormaps import cm
from aiapy.calibrate import prep


def load_KSO(f):
    """
    Load a KSO H-alpha FITS file and return a SunPy Map.
    """
    kso_map = Map(f)
    angle = -kso_map.meta["angle"]

    kso_map.meta["waveunit"] = "AA"
    kso_map.meta["arcs_pp"] = kso_map.scale[0].value

    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))

    kso_map.meta["PC1_1"] = c
    kso_map.meta["PC1_2"] = -s
    kso_map.meta["PC2_1"] = s
    kso_map.meta["PC2_2"] = c
    return kso_map

pre_eruption_map = load_KSO('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/kanz_halph_fi_20230806_085110.fts.gz')
pre_eruption_map = pre_eruption_map.rotate()

coords = all_coordinates_from_map(pre_eruption_map).transform_to(frames.Helioprojective)
radius = np.sqrt(coords.Tx**2 + coords.Ty**2) / pre_eruption_map.rsun_obs
data = pre_eruption_map.data.astype(np.float64)
data[radius >= 1] = np.nan  # Mask out pixels outside the radius of 0.9 Rsun

carrington_center = SkyCoord(lat=10 * u.deg, lon=30 * u.deg, frame=frames.HeliographicCarrington,
                             obstime=pre_eruption_map.date, observer=pre_eruption_map.observer_coordinate)

##### full map ####
fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(data, cmap='gray', origin='lower')

ax.set_axis_off()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/full_0851.png', dpi=300, transparent=True)
plt.close()

#### submap ####
hpc_center = carrington_center.transform_to(frames.Helioprojective(observer=pre_eruption_map.observer_coordinate))
sub_map_pre = pre_eruption_map.submap(hpc_center, width=256 * u.arcsec, height=150 * u.arcsec)

fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': sub_map_pre})
ax.imshow(sub_map_pre.data, cmap='gray', origin='lower')

ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')
fig.tight_layout()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/submap_0851.png', dpi=300, transparent=True)
plt.close()

######## post eruption map ########
post_eruption_map = load_KSO('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/kanz_halph_fi_20230806_104913.fts.gz')

hpc_center = carrington_center.transform_to(frames.Helioprojective(observer=post_eruption_map.observer_coordinate))
sub_map_post = post_eruption_map.submap(hpc_center, width=256 * u.arcsec, height=150 * u.arcsec)

fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': sub_map_post})
ax.imshow(sub_map_post.data, cmap='gray', origin='lower')

ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')
fig.tight_layout()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/submap_1049.png', dpi=300, transparent=True)
plt.close()

###########################################
# visualize EUV

euv_map = Map('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/aia.lev1_euv_12s.2023-08-06T085111Z.193.image_lev1.fits')

coords = all_coordinates_from_map(euv_map).transform_to(frames.Helioprojective)
radius = np.sqrt(coords.Tx**2 + coords.Ty**2) / pre_eruption_map.rsun_obs
data = euv_map.data.astype(np.float64)
mask = np.clip(1 - (radius.value - 1) / 0.3, a_min=0, a_max=1)
mask[radius <= 1] = 1

fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(data, cmap=cm.sdoaia193, origin='lower', alpha=mask, norm='log', vmin=100)

ax.set_axis_off()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/euv_full_0851.png', dpi=300, transparent=True)
plt.close()

# Submap for EUV
bottom_left = SkyCoord(200 * u.arcsec, 0 * u.arcsec, frame=euv_map.coordinate_frame)
sub_map_euv = euv_map.submap(bottom_left, width=256 * u.arcsec, height=150 * u.arcsec)

fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': sub_map_euv})
ax.imshow(sub_map_euv.data, cmap=cm.sdoaia193, origin='lower', norm='log', vmin=100)
ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')
fig.tight_layout()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/euv_submap_0851.png', dpi=300, transparent=True)
plt.close()

# EUV post eruption

euv_map_post = Map('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/aia.lev1_euv_12s.2023-08-06T105006Z.193.image_lev1.fits')

coords = all_coordinates_from_map(euv_map_post).transform_to(frames.Helioprojective)
radius = np.sqrt(coords.Tx**2 + coords.Ty**2) / pre_eruption_map.rsun_obs
data = euv_map_post.data.astype(np.float64)
mask = np.clip(1 - (radius.value - 1) / 0.3, a_min=0, a_max=1)
mask[radius <= 1] = 1

fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(data, cmap=cm.sdoaia193, origin='lower', alpha=mask, norm='log', vmin=100)

ax.set_axis_off()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/2023_08_06/euv_full_1050.png', dpi=300, transparent=True)
plt.close()



##################
# plot BLOS panorama
data = fits.getdata('/Users/rjarolim/PycharmProjects/NF2/data/campaign_2023_v2/BLOS_panorama0851_6173_StkIQUV_rebin.fits')

extent = [0, data.shape[1] * 0.09, 0, data.shape[0] * 0.09]

fig, ax = plt.subplots(figsize=(4, 3))

im =ax.imshow(data, cmap='gray', origin='lower', extent=extent, vmin=-1000, vmax=1000)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)

ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')
fig.tight_layout()
fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/campaign_2023_v2/BLOS_panorama0851_6173_StkIQUV_rebin.png', dpi=300, transparent=True)
plt.close()
