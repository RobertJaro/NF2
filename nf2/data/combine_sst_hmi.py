import copy

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from astropy import units as u

hmi_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/7310/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Br.fits')

data = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
meta = fits.getheader('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
meta['crota2'] = float(meta['crota2'])
meta['CTYPE1'] = 'CRLN-CEA'
meta['CTYPE2'] = 'CRLT-CEA'
meta['CRVAL1'] = meta['CRVAL1'] - .1
meta['CRVAL2'] = meta['CRVAL2']

sst_map = Map(data[2], meta)

# convert to Helioprojective
hmi_coords = all_coordinates_from_map(hmi_map).transform_to(frames.Helioprojective)
sst_coords = all_coordinates_from_map(sst_map).transform_to(frames.Helioprojective)

lims = np.array([
    [min(hmi_coords.Tx.min().value, sst_coords.Tx.min().value), max(hmi_coords.Tx.max().value, sst_coords.Tx.max().value)],
    [min(hmi_coords.Ty.min().value, sst_coords.Ty.min().value), max(hmi_coords.Ty.max().value, sst_coords.Ty.max().value)]])


bottom_left = SkyCoord(lims[0, 0] * u.arcsec, lims[1, 0] * u.arcsec, frame=frames.Helioprojective, observer=sst_map.observer_coordinate)
top_right = SkyCoord(lims[0, 1] * u.arcsec, lims[1, 1] * u.arcsec, frame=frames.Helioprojective, observer=sst_map.observer_coordinate)

bottom_left_pixel = sst_map.world_to_pixel(bottom_left)
top_right_pixel = sst_map.world_to_pixel(top_right)


shape = (top_right_pixel.y - bottom_left_pixel.y, top_right_pixel.x - bottom_left_pixel.x)
shape = (int(shape[0].value), int(shape[1].value))

target_wcs = copy.deepcopy(sst_map.wcs)
target_wcs.array_shape = shape

ref_pixel = sst_map.reference_pixel
ref_bottom_left = sst_map.world_to_pixel(sst_map.bottom_left_coord)
ref_top_right = sst_map.world_to_pixel(sst_map.top_right_coord)
target_wcs.wcs.crpix = [ref_pixel.x.value + (ref_bottom_left.x.value - bottom_left_pixel.x.value),
                 ref_pixel.y.value + (ref_bottom_left.y.value - bottom_left_pixel.y.value)]

#### SAVE FITS ####

mask = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea_mask.fits')

data = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_8542_2018-09-30_mosaic_Bvec_cea.fits')
data[:, mask == 0] = np.nan
meta = fits.getheader('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_8542_2018-09-30_mosaic_Bvec_cea.fits')
meta['crota2'] = float(meta['crota2'])
meta['CTYPE1'] = 'CRLN-CEA'
meta['CTYPE2'] = 'CRLT-CEA'
meta['CRVAL1'] = meta['CRVAL1'] - .1
meta['CRVAL2'] = meta['CRVAL2']
for i, l in enumerate(['p', 't', 'r']):
    sst_map = Map(data[i], meta).reproject_to(target_wcs)
    sst_map.save(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_8542_2018-09-30_B{l}_cea.fits', overwrite=True)
    sst_map.plot(vmin=-1000, vmax=1000)
    plt.savefig(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_8542_2018-09-30_B{l}_cea.jpg')
    plt.close()


data = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
data[:, mask == 0] = np.nan
meta = fits.getheader('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
meta['crota2'] = float(meta['crota2'])
meta['CTYPE1'] = 'CRLN-CEA'
meta['CTYPE2'] = 'CRLT-CEA'
meta['CRVAL1'] = meta['CRVAL1'] - .1
meta['CRVAL2'] = meta['CRVAL2']
for i, l in enumerate(['p', 't', 'r']):
    sst_map = Map(data[i], meta).reproject_to(target_wcs)
    sst_map.save(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_6173_2018-09-30_B{l}_cea.fits', overwrite=True)
    sst_map.plot(vmin=-1000, vmax=1000)
    plt.savefig(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_6173_2018-09-30_B{l}_cea.jpg')
    plt.close()


for i, l in enumerate(['p', 't', 'r']):
    hmi_map = Map(f'/gpfs/gpfs0/robert.jarolim/data/nf2/7310/hmi.sharp_cea_720s.7310.20180930_092400_TAI.B{l}.fits')
    hmi_map = hmi_map.reproject_to(target_wcs)
    hmi_map.data[~np.isnan(sst_map.data)] = np.nan
    hmi_map.save(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_hmi_2018-09-30_B{l}_cea.fits', overwrite=True)
    hmi_map.plot(vmin=-1000, vmax=1000)
    plt.savefig(f'/gpfs/gpfs0/robert.jarolim/data/nf2/sst/corrected_hmi_2018-09-30_B{l}_cea.jpg')
    plt.close()

# PLOT overview
hmi_p_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/7310/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Bp.fits')
hmi_t_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/7310/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Bt.fits')
hmi_r_map = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/7310/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Br.fits')

data = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
meta = fits.getheader('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_6173_2018-09-30_mosaic_Bvec_cea.fits')
meta['crota2'] = float(meta['crota2'])
meta['CTYPE1'] = 'CRLN-CEA'
meta['CTYPE2'] = 'CRLT-CEA'
meta['CRVAL1'] = meta['CRVAL1'] - .1
meta['CRVAL2'] = meta['CRVAL2']
sst_6173_p_map = Map(data[0], meta)
sst_6173_t_map = Map(data[1], meta)
sst_6173_r_map = Map(data[2], meta)


data = fits.getdata('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_8542_2018-09-30_mosaic_Bvec_cea.fits')
meta = fits.getheader('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/nb_8542_2018-09-30_mosaic_Bvec_cea.fits')
meta['crota2'] = float(meta['crota2'])
meta['CTYPE1'] = 'CRLN-CEA'
meta['CTYPE2'] = 'CRLT-CEA'
meta['CRVAL1'] = meta['CRVAL1'] - .1
meta['CRVAL2'] = meta['CRVAL2']
sst_8542_p_map = Map(data[0], meta)
sst_8542_t_map = Map(data[1], meta)
sst_8542_r_map = Map(data[2], meta)


fig, axs = plt.subplots(3, 3, figsize=(15, 15))

axs[0, 0].imshow(hmi_p_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                 extent=[hmi_coords.Tx.min().value, hmi_coords.Tx.max().value, hmi_coords.Ty.min().value, hmi_coords.Ty.max().value])

axs[0, 1].imshow(hmi_t_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[hmi_coords.Tx.min().value, hmi_coords.Tx.max().value, hmi_coords.Ty.min().value, hmi_coords.Ty.max().value])

axs[0, 2].imshow(hmi_r_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[hmi_coords.Tx.min().value, hmi_coords.Tx.max().value, hmi_coords.Ty.min().value, hmi_coords.Ty.max().value])

axs[1, 0].imshow(sst_6173_p_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

axs[1, 1].imshow(sst_6173_t_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

axs[1, 2].imshow(sst_6173_r_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

axs[2, 0].imshow(sst_8542_p_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

axs[2, 1].imshow(sst_8542_t_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

axs[2, 2].imshow(sst_8542_r_map.data, origin='lower', cmap='gray', vmin=-1500, vmax=1500,
                    extent=[sst_coords.Tx.min().value, sst_coords.Tx.max().value, sst_coords.Ty.min().value, sst_coords.Ty.max().value])

[ax.set_xlim(lims[0, 0], lims[0, 1]) for ax in axs.flatten()]
[ax.set_ylim(lims[1, 0], lims[1, 1]) for ax in axs.flatten()]

plt.tight_layout()
plt.savefig('/gpfs/gpfs0/robert.jarolim/data/nf2/sst/overview.jpg')
plt.close()

# padded map
bottom_left_coord = SkyCoord(lims[0] * u.arcsec, lims[2] * u.arcsec, frame=hmi_coords.frame)
top_right_coord = SkyCoord(lims[1] * u.arcsec, lims[3] * u.arcsec, frame=hmi_coords.frame)
hmi_map = hmi_map.submap(bottom_left=bottom_left_coord, top_right=top_right_coord)

