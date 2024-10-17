import glob
import os

from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

out_path = '/glade/work/rjarolim/nf2/sst/evaluation'
os.makedirs(out_path, exist_ok=True)

los_files = '/glade/work/rjarolim/data/SST/campaign_2023_v2/BLOS*.fits'
trv_files = '/glade/work/rjarolim/data/SST/campaign_2023_v2/BTRV*.fits'
azi_files = '/glade/work/rjarolim/data/SST/campaign_2023_v2/BAZI*.fits'
mask = fits.getdata('/glade/work/rjarolim/data/SST/campaign_2023_v2/mask.fits')

fig, axs = plt.subplots(5, 3, figsize=(13, 10))

labels = [r'5896 $\text{\AA}$', '6173 $\AA$', '7699 $\AA$', '7699 v2 $\AA$', '8542 $\AA$']

# Plot LOS
for i in range(5):
    ax = axs[i, 0]
    data = fits.getdata(los_files[i])
    data[mask == 1] = np.nan
    im = ax.imshow(data, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'LOS {labels[i]}')
    # Plot TRV
    ax = axs[i, 1]
    data = fits.getdata(trv_files[i])
    data[mask == 1] = np.nan
    im = ax.imshow(data, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'TRV {labels[i]}')
    # Plot AZI
    ax = axs[i, 2]
    data = fits.getdata(azi_files[i])
    data[mask == 1] = np.nan
    im = ax.imshow(data % np.pi, cmap='twilight', origin='lower', vmin=0, vmax=np.pi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(f'AZI {labels[i]}')

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_data_overview.png'))
plt.show()