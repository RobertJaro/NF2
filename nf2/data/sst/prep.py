import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header

hmi_field = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.field.fits')
hmi_azi = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.azimuth.fits')
hmi_inc = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.inclination.fits')
hmi_disamb = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.disambig.fits')

sst_los_f = sorted(glob.glob('/glade/work/rjarolim/data/SST/BLOS*.fits'))
sst_trv_f = sorted(glob.glob('/glade/work/rjarolim/data/SST/BTRV*.fits'))
sst_azi_f = sorted(glob.glob('/glade/work/rjarolim/data/SST/BAZI*.fits'))


for f_los, f_trv, f_azi in zip(sst_los_f, sst_trv_f, sst_azi_f):
    sst_los_map = Map(f_los)
    sst_trv_map = Map(f_trv)
    sst_azi_map = Map(f_azi)

    ref_pix = (sst_los_map.reference_pixel.x.value, sst_los_map.reference_pixel.y.value) * u.pix
    scale = (0.0075, 0.0075) * u.deg / u.pix
    cea_header = make_fitswcs_header(sst_los_map.data,
                                     sst_los_map.reference_coordinate.transform_to(frames.HeliographicCarrington),
                                     ref_pix, scale=scale, projection_code='CEA')

    cea_sst_los_map = sst_los_map.reproject_to(cea_header)
    cea_sst_trv_map = sst_trv_map.reproject_to(cea_header)
    cea_sst_azi_map = sst_azi_map.reproject_to(cea_header)

    mask = np.isnan(cea_sst_los_map.data) | (cea_sst_azi_map.data == 0)

    cea_sst_los_map.data[mask] = np.nan
    cea_sst_trv_map.data[mask] = np.nan
    cea_sst_azi_map.data[mask] = np.nan

    cea_sst_los_map.save(os.path.join('/glade/work/rjarolim/data/SST/converted', os.path.basename(f_los)), overwrite=True)
    cea_sst_trv_map.save(os.path.join('/glade/work/rjarolim/data/SST/converted', os.path.basename(f_trv)), overwrite=True)
    cea_sst_azi_map.save(os.path.join('/glade/work/rjarolim/data/SST/converted', os.path.basename(f_azi)), overwrite=True)

    fig = plt.figure(figsize=(10, 10))

    ax = plt.subplot(3, 2, 1, projection=cea_sst_los_map)
    sst_los_map.plot(axes=ax, cmap='gray')
    ax.set_title('LOS')
    plt.colorbar()

    ax = plt.subplot(3, 2, 2, projection=cea_sst_los_map)
    cea_sst_los_map.plot(axes=ax, cmap='gray')
    ax.set_title('LOS CEA')
    plt.colorbar()

    ax = plt.subplot(3, 2, 3, projection=cea_sst_trv_map)
    sst_trv_map.plot(axes=ax, cmap='gray')
    ax.set_title('TRV')
    plt.colorbar()

    ax = plt.subplot(3, 2, 4, projection=cea_sst_trv_map)
    cea_sst_trv_map.plot(axes=ax, cmap='gray')
    ax.set_title('TRV CEA')
    plt.colorbar()

    ax = plt.subplot(3, 2, 5, projection=cea_sst_azi_map)
    sst_azi_map.plot(axes=ax, cmap='twilight')
    ax.set_title('AZI')
    plt.colorbar()

    ax = plt.subplot(3, 2, 6, projection=cea_sst_azi_map)
    cea_sst_azi_map.plot(axes=ax, cmap='twilight')
    ax.set_title('AZI CEA')
    plt.colorbar()

    fig.tight_layout()
    fig.savefig(os.path.join('/glade/work/rjarolim/data/SST/converted/', os.path.basename(f_los).replace('.fits', '.jpg')))
    plt.close(fig)

cea_hmi_field = hmi_field.reproject_to(sst_los_map.wcs).reproject_to(cea_header)
cea_hmi_azi = hmi_azi.reproject_to(sst_los_map.wcs).reproject_to(cea_header)
cea_hmi_inc = hmi_inc.reproject_to(sst_los_map.wcs).reproject_to(cea_header)
cea_hmi_disamb = hmi_disamb.reproject_to(sst_los_map.wcs).reproject_to(cea_header)

field = np.nan_to_num(cea_hmi_field.data, nan=0)
azi = np.nan_to_num(np.deg2rad(cea_hmi_azi.data), nan=0)
inc = np.nan_to_num(np.deg2rad(cea_hmi_inc.data), nan=0)
amb = cea_hmi_disamb.data


mask = ~np.isnan(cea_sst_los_map.data)

cea_hmi_los = Map(field * np.cos(inc), cea_header)
cea_hmi_los_masked = Map(field * np.cos(inc), cea_header)
cea_hmi_los_masked.data[mask] = np.nan
cea_hmi_los_masked.save('/glade/work/rjarolim/data/SST/converted/hmi_los_masked.fits', overwrite=True)
cea_hmi_los.save('/glade/work/rjarolim/data/SST/converted/hmi_los.fits', overwrite=True)

cea_hmi_trv = Map(field * np.sin(inc), cea_header)
cea_hmi_trv_masked = Map(field * np.sin(inc), cea_header)
cea_hmi_trv_masked.data[mask] = np.nan
cea_hmi_trv_masked.save('/glade/work/rjarolim/data/SST/converted/hmi_trv_masked.fits', overwrite=True)
cea_hmi_trv.save('/glade/work/rjarolim/data/SST/converted/hmi_trv.fits', overwrite=True)

amb_weak = 2
condition = (amb.astype((int)) >> amb_weak).astype(bool)
azi[condition] += np.pi

cea_hmi_azi = Map(azi, cea_header)
cea_hmi_azi_masked = Map(np.copy(azi), cea_header)
cea_hmi_azi_masked.data[mask] = np.nan
cea_hmi_azi_masked.save('/glade/work/rjarolim/data/SST/converted/hmi_azi_masked.fits', overwrite=True)
cea_hmi_azi.save('/glade/work/rjarolim/data/SST/converted/hmi_azi.fits', overwrite=True)

fig = plt.figure(figsize=(10, 10))

ax = plt.subplot(4, 2, 1, projection=cea_hmi_los)
cea_hmi_los.plot(axes=ax, cmap='gray')
ax.set_title('LOS')
plt.colorbar()

ax = plt.subplot(4, 2, 2, projection=cea_hmi_los_masked)
cea_hmi_los_masked.plot(axes=ax, cmap='gray')
ax.set_title('LOS MASKED')
plt.colorbar()

ax = plt.subplot(4, 2, 3, projection=cea_hmi_trv)
cea_hmi_trv.plot(axes=ax, cmap='gray')
ax.set_title('TRV')
plt.colorbar()

ax = plt.subplot(4, 2, 4, projection=cea_hmi_trv_masked)
cea_hmi_trv_masked.plot(axes=ax, cmap='gray')
ax.set_title('TRV MASKED')
plt.colorbar()

ax = plt.subplot(4, 2, 5, projection=cea_hmi_azi)
cea_hmi_azi.plot(axes=ax, cmap='twilight', vmin=0, vmax=2 * np.pi)
ax.set_title('AZI')
plt.colorbar()

ax = plt.subplot(4, 2, 6, projection=cea_hmi_azi_masked)
cea_hmi_azi_masked.plot(axes=ax, cmap='twilight', vmin=0, vmax=2 * np.pi)
ax.set_title('AZI MASKED')
plt.colorbar()

ax = plt.subplot(4, 2, 7, projection=cea_hmi_azi)
ax.imshow(cea_hmi_azi.data % np.pi, axes=ax, cmap='twilight', vmin=0, vmax=np.pi)
ax.set_title('AZI AMB')
plt.colorbar()

ax = plt.subplot(4, 2, 8, projection=cea_hmi_azi_masked)
ax.imshow(cea_hmi_azi_masked.data % np.pi, axes=ax, cmap='twilight', vmin=0, vmax=np.pi)
ax.set_title('AZI AMB MASKED')
plt.colorbar()

fig.tight_layout()
fig.savefig('/glade/work/rjarolim/data/SST/converted/hmi.jpg')
plt.close(fig)
