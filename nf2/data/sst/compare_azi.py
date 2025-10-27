import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sst_azi = '/glade/work/rjarolim/data/nf2/nlte/BAZI_2018-09-30T09:22:00_NLTE_ltau_-0.0.fits'

hmi_azi = '/glade/work/rjarolim/data/nf2/12723/hmi.sharp_720s.7310.20180930_092400_TAI.azimuth.fits'
hmi_inc = '/glade/work/rjarolim/data/nf2/12723/hmi.sharp_720s.7310.20180930_092400_TAI.inclination.fits'
hmi_fld = '/glade/work/rjarolim/data/nf2/12723/hmi.sharp_720s.7310.20180930_092400_TAI.field.fits'
hmi_disamb = '/glade/work/rjarolim/data/nf2/12723/hmi.sharp_720s.7310.20180930_092400_TAI.disambig.fits'

data_sst_azi = fits.getdata(sst_azi) % np.pi

data_hmi_azi = np.deg2rad(fits.getdata(hmi_azi)) % np.pi
amb = fits.getdata(hmi_disamb)
amb_weak = 2
condition = (amb.astype(int) >> amb_weak).astype(bool)

data_hmi_azi[condition] += np.pi
data_hmi_inc = np.deg2rad(fits.getdata(hmi_inc))
data_hmi_fld = fits.getdata(hmi_fld)
data_hmi_azi = np.flip(data_hmi_azi, axis=(0, 1))
data_hmi_inc = np.flip(data_hmi_inc, axis=(0, 1))
data_hmi_fld = np.flip(data_hmi_fld, axis=(0, 1))

hmi_bx = data_hmi_fld * np.sin(data_hmi_inc) * np.sin(np.pi - data_hmi_azi)
hmi_by = data_hmi_fld * np.sin(data_hmi_inc) * np.cos(np.pi - data_hmi_azi)
hmi_bz = data_hmi_fld * np.cos(data_hmi_inc)

fix, axs = plt.subplots(1, 3, figsize=(15, 5))
ax = axs[0]
im = ax.imshow(hmi_bx, vmin=-1500, vmax=1500, cmap='gray', origin='lower')
ax.set_title('HMI Bx')
plt.colorbar(im, ax=ax)
ax = axs[1]
im = ax.imshow(hmi_by, vmin=-1500, vmax=1500,
                cmap='gray', origin='lower')
ax.set_title('HMI By')
plt.colorbar(im, ax=ax)
ax = axs[2]
im = ax.imshow(hmi_bz, vmin=-1500, vmax=1500
                , cmap='gray', origin='lower')
ax.set_title('HMI Bz')
plt.colorbar(im, ax=ax)
plt.savefig('/glade/work/rjarolim/data/nf2/nlte/hmi_bvec_test.jpg', dpi=200)
plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
im = ax.imshow(data_sst_azi, vmin=0, vmax=np.pi,
               cmap='twilight', origin='lower')
ax.set_title('SST Azimuth')
plt.colorbar(im, ax=ax)
ax = axs[1]
im = ax.imshow(data_hmi_azi % np.pi, vmin=0, vmax=np.pi,
               cmap='twilight', origin='lower')
ax.set_title('HMI Azimuth')
plt.colorbar(im, ax=ax)
plt.savefig('/glade/work/rjarolim/data/nf2/nlte/azi_test.jpg', dpi=200)
plt.close()
