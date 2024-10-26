import glob
import os

from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

out_path = '/glade/work/rjarolim/nf2/sst/evaluation'
os.makedirs(out_path, exist_ok=True)

wl_keys = ['6173_StkIQUV_rebin', '7699_StkIQUV_rebin_v2', '8542_StkIQUV_rebin']

los_files = [f'/glade/work/rjarolim/data/SST/campaign_2023_v2/BLOS_panorama0851_{i}.fits' for i in wl_keys]
trv_files = [f'/glade/work/rjarolim/data/SST/campaign_2023_v2/BTRV_panorama0851_{i}.fits' for i in wl_keys]
azi_files = [f'/glade/work/rjarolim/data/SST/campaign_2023_v2/BAZI_panorama0851_{i}.fits' for i in wl_keys]
mask = fits.getdata('/glade/work/rjarolim/data/SST/campaign_2023_v2/mask.fits')

fig, axs = plt.subplots(len(wl_keys), 4, figsize=(13, 6))

labels = ['6173 Å', '7699 Å' , '8542 Å']

# Plot LOS
for i in range(len(labels)):
    ax = axs[i, 0]
    data_los = fits.getdata(los_files[i])
    data_los[mask == 1] = np.nan
    im = ax.imshow(data_los, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'LOS {labels[i]}')
    # Plot TRV
    ax = axs[i, 1]
    data_trv = fits.getdata(trv_files[i])
    data_trv[mask == 1] = np.nan
    im = ax.imshow(data_trv, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'TRV {labels[i]}')
    # Plot AZI
    ax = axs[i, 2]
    data_azi = fits.getdata(azi_files[i])
    data_azi[mask == 1] = np.nan
    im = ax.imshow(data_azi % np.pi, cmap='twilight', origin='lower', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'AZI {labels[i]}')
    #
    ax = axs[i, 3]
    b_abs = np.sqrt(data_los**2 + data_trv**2)
    ax.hist(b_abs.flatten(), bins=100, range=(0, 1000))
    ax.set_title(r'$\left\Vert B \right\Vert$ histogram ' + f'{labels[i]}')
    # add mean
    mean_b = np.nanmean(b_abs)
    mean_b_los = np.nanmean(np.abs(data_los))
    ax.axvline(mean_b, color='red', linestyle='--', label=r'$\overline{\left\Vert B \right\Vert}$: ' + f'{mean_b:.0f} G')
    ax.axvline(mean_b_los, color='blue', linestyle='--', label=r'$\overline{\left\Vert B_\text{LOS} \right\Vert}$: ' + f'{mean_b_los:.0f} G')
    ax.axvline(200, color='black', linestyle='--', label=r'$\left\Vert B \right\Vert = 200$ G', alpha=0.5)
    ax.legend()
    ax.set_ylim(0, 1.5e5)

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_data_overview.png'))
plt.show()

########################################################################################################################
# scatter plots

b0_mag = np.sqrt(fits.getdata(los_files[0])**2 + fits.getdata(trv_files[0])**2)
b1_mag = np.sqrt(fits.getdata(los_files[1])**2 + fits.getdata(trv_files[1])**2)
b2_mag = np.sqrt(fits.getdata(los_files[2])**2 + fits.getdata(trv_files[2])**2)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax = axs[0]
# 2d histogram
ax.hist2d(b0_mag.flatten(), b1_mag.flatten(), bins=100, range=[[0, 3000], [0, 3000]], cmap='viridis', norm='log')
ax.set_xlabel(r'$\left\Vert B \right\Vert$ 6173 Å [G]')
ax.set_ylabel(r'$\left\Vert B \right\Vert$ 7699 Å [G]')


ax = axs[1]
ax.hist2d(b0_mag.flatten(), b2_mag.flatten(), bins=100, range=[[0, 3000], [0, 3000]], cmap='viridis', norm='log')
ax.set_xlabel(r'$\left\Vert B \right\Vert$ 6173 Å [G]')
ax.set_ylabel(r'$\left\Vert B \right\Vert$ 8542 Å [G]')

for ax in axs:
    ax.plot([0, 3000], [0, 3000], color='black', linestyle='--')
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 3000)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_scatter.png'))
plt.close()

