import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

from nf2.evaluation.kso.util import load_KSO

# pre-eruption maps
br_pre_map = Map('/Users/rjarolim/PycharmProjects/NF2/data/topology/hmi.B_720s.20230806_091200_TAI.Br.fits')
kso_pre_map = load_KSO('/Users/rjarolim/PycharmProjects/NF2/data/topology/kanz_halph_fi_20230806_085003.fts.gz')
aia_pre_map = Map(
    '/Users/rjarolim/PycharmProjects/NF2/data/topology/aia.lev1_euv_12s.2023-08-06T085209Z.171.image_lev1.fits')
aia_304_pre_map = Map('/Users/rjarolim/PycharmProjects/NF2/data/topology/aia.lev1_euv_12s.2023-08-06T085507Z.304.image_lev1.fits')

# post-eruption maps
br_post_map = Map('/Users/rjarolim/PycharmProjects/NF2/data/topology/hmi.B_720s.20230806_104800_TAI.Br.fits')
kso_post_map = load_KSO('/Users/rjarolim/PycharmProjects/NF2/data/topology/kanz_halph_fi_20230806_105321.fts.gz')
aia_post_map = Map(
    '/Users/rjarolim/PycharmProjects/NF2/data/topology/aia.lev1_euv_12s.2023-08-06T105010Z.171.image_lev1.fits')
aia_304_post_map = Map('/Users/rjarolim/PycharmProjects/NF2/data/topology/aia.lev1_euv_12s.2023-08-06T105007Z.304.image_lev1.fits')

# exposure time correction for all maps
data = aia_pre_map.data / aia_pre_map.exposure_time.to_value(u.s)
aia_pre_map = Map(data, aia_pre_map.meta)
data = aia_post_map.data / aia_post_map.exposure_time.to_value(u.s)
aia_post_map = Map(data, aia_post_map.meta)
data = aia_304_pre_map.data / aia_304_pre_map.exposure_time.to_value(u.s)
aia_304_pre_map = Map(data, aia_304_pre_map.meta)
data = aia_304_post_map.data / aia_304_post_map.exposure_time.to_value(u.s)
aia_304_post_map = Map(data, aia_304_post_map.meta)
data = kso_pre_map.data / kso_pre_map.exposure_time.to_value(u.s)
kso_pre_map = Map(data, kso_pre_map.meta)
data = kso_post_map.data / kso_post_map.exposure_time.to_value(u.s)
kso_post_map = Map(data, kso_post_map.meta)


# create submap centered on the eruption site
def _crop_submap(s_map):
    bottom_left_coord = SkyCoord(260 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective,
                                 observer=s_map.observer_coordinate)
    top_right_coord = SkyCoord(490 * u.arcsec, 140 * u.arcsec, frame=frames.Helioprojective,
                               observer=s_map.observer_coordinate)
    return s_map.submap(bottom_left=bottom_left_coord, top_right=top_right_coord)

def _get_extent(s_map):
    return [s_map.bottom_left_coord.Tx.to_value(u.arcsec), s_map.top_right_coord.Tx.to_value(u.arcsec),
            s_map.bottom_left_coord.Ty.to_value(u.arcsec), s_map.top_right_coord.Ty.to_value(u.arcsec)]

aia_pre_submap = _crop_submap(aia_pre_map)
aia_post_submap = aia_post_map.reproject_to(aia_pre_submap.wcs) #_crop_submap(aia_post_map)
aia_304_pre_submap = aia_304_pre_map.reproject_to(aia_pre_submap.wcs)
aia_304_post_submap = aia_304_post_map.reproject_to(aia_post_submap.wcs)
kso_pre_submap = kso_pre_map.reproject_to(aia_pre_submap.wcs)
kso_post_submap  = kso_post_map.reproject_to(aia_pre_submap.wcs)
br_pre_submap  = br_pre_map.reproject_to(aia_pre_submap.wcs)
br_post_submap  = br_post_map.reproject_to(aia_pre_submap.wcs)

# crop overview map
bottom_left_coord = SkyCoord(-100 * u.arcsec, -100 * u.arcsec, frame=frames.Helioprojective,
                             observer=aia_pre_map.observer_coordinate)
top_right_coord = SkyCoord(1050 * u.arcsec, 1000 * u.arcsec, frame=frames.Helioprojective,
                           observer=aia_pre_map.observer_coordinate)
aia_pre_map = aia_pre_map.submap(bottom_left=bottom_left_coord, top_right=top_right_coord)

# faint off limb region
coords = all_coordinates_from_map(aia_pre_map).transform_to(frames.Helioprojective)
radius = np.sqrt(coords.Tx**2 + coords.Ty**2) / aia_pre_map.rsun_obs
euv_mask = np.clip(1 - (radius.value - 1) / 0.3, a_min=0, a_max=1) ** 4
euv_mask[radius <= 1] = 1

# constraint layout with one full-disc on top, three columns and two rows below (AIA, Halpha, colorbar; Pre-eruption, Post-eruption)
kso_norm = LogNorm(vmin=2.5e5)
aia_norm = LogNorm(vmin=20)
aia_304_norm = LogNorm(vmin=1)

fig = plt.figure(figsize=(8, 14), dpi=300)
gs = fig.add_gridspec(nrows=5, ncols=3,
                      height_ratios=[3, 0.1, 1, 1, 1], width_ratios=[1, 1, 0.03],
                      hspace=0.1, wspace=0.1)

# full disc
ax_full = fig.add_subplot(gs[0, :-1])
extent = _get_extent(aia_pre_map)
im = ax_full.imshow(aia_pre_map.data, cmap=cm.sdoaia171, origin='lower', alpha=euv_mask, norm=aia_norm, extent=extent)
# hide top and right spines
ax_full.spines['top'].set_visible(False)
ax_full.spines['right'].set_visible(False)

ax_full.set_xlabel('Helioprojective X [arcsec]')
ax_full.set_ylabel('Helioprojective Y [arcsec]')
# add colorbar for full disc
cax_full = fig.add_subplot(gs[0, -1])
cbar_full = fig.colorbar(im, cax=cax_full)
cbar_full.set_label(r'AIA 171 $\rm\AA$ [DN/s]')

# AIA - pre eruption + HMI contours
ax_aia_pre = fig.add_subplot(gs[2, 0])
extent = _get_extent(aia_pre_submap)
aia_im = ax_aia_pre.imshow(aia_pre_submap.data, cmap=cm.sdoaia171, origin='lower', norm=aia_norm, extent=extent)
ax_aia_pre.contour(br_pre_submap.data, levels=[-500, 500], colors=['blue', 'red'], linewidths=0.8, extent=extent, alpha=0.7)
ax_aia_pre.set_xlabel(' ')
ax_aia_pre.set_ylabel('Helioprojective Y [arcsec]')
ax_aia_pre.set_xticklabels([])
# AIA - post eruption + HMI contours
ax_aia_post = fig.add_subplot(gs[2, 1])
extent = _get_extent(aia_post_submap)
ax_aia_post.imshow(aia_post_submap.data, cmap=cm.sdoaia171, origin='lower', norm=aia_norm, extent=extent)
ax_aia_post.contour(br_post_submap.data, levels=[-500, 500], colors=['blue', 'red'], linewidths=0.8, extent=extent, alpha=0.7)
ax_aia_post.set_xlabel(' ')
ax_aia_post.set_ylabel(' ')
ax_aia_post.set_yticklabels([])
ax_aia_post.set_xticklabels([])
# AIA colorbar
cax_aia = fig.add_subplot(gs[2, 2])
cbar_aia = fig.colorbar(aia_im, cax=cax_aia)
cbar_aia.set_label(r'AIA 171 $\rm\AA$ [DN/s]')

# AIA 304 - pre eruption
ax_aia_304_pre = fig.add_subplot(gs[3, 0])
extent = _get_extent(aia_304_pre_submap)
aia_304_im = ax_aia_304_pre.imshow(aia_304_pre_submap.data, cmap=cm.sdoaia304, origin='lower', norm=aia_304_norm, extent=extent)
ax_aia_304_pre.set_xlabel(' ')
ax_aia_304_pre.set_ylabel('Helioprojective Y [arcsec]')
ax_aia_304_pre.set_xticklabels([])
# AIA 304 - post eruption
ax_aia_304_post = fig.add_subplot(gs[3, 1])
extent = _get_extent(aia_304_post_submap)
ax_aia_304_post.imshow(aia_304_post_submap.data, cmap=cm.sdoaia304, origin='lower', norm=aia_304_norm, extent=extent)
ax_aia_304_post.set_xlabel(' ')
ax_aia_304_post.set_ylabel(' ')
ax_aia_304_post.set_yticklabels([])
ax_aia_304_post.set_xticklabels([])
# AIA 304 colorbar
cax_aia_304 = fig.add_subplot(gs[3, 2])
cbar_aia_304 = fig.colorbar(aia_304_im, cax=cax_aia_304)
cbar_aia_304.set_label(r'AIA 304 $\rm\AA$ [DN/s]')

# KSO - pre eruption
ax_kso_pre = fig.add_subplot(gs[4, 0])
extent = _get_extent(kso_pre_submap)
kso_pre_im = ax_kso_pre.imshow(kso_pre_submap.data, cmap='gray', origin='lower', norm=kso_norm, extent=extent)
ax_kso_pre.set_xlabel('Helioprojective X [arcsec]')
ax_kso_pre.set_ylabel('Helioprojective Y [arcsec]')
# KSO - post eruption
ax_kso_post = fig.add_subplot(gs[4, 1])
extent = _get_extent(kso_post_submap)
ax_kso_post.imshow(kso_post_submap.data, cmap='gray', origin='lower', norm=kso_norm, extent=extent)
ax_kso_post.set_xlabel('Helioprojective X [arcsec]')
ax_kso_post.set_ylabel(' ')
ax_kso_post.set_yticklabels([])
# KSO colorbar
cax_kso = fig.add_subplot(gs[4, 2])
cbar_kso = fig.colorbar(kso_pre_im, cax=cax_kso)
cbar_kso.set_label(r'KSO H$\alpha$ [DN/s]')

fig.savefig('/Users/rjarolim/PycharmProjects/NF2/data/topology/observations_comparison.png', dpi=300, transparent=True)
plt.close(fig)
fig.show()