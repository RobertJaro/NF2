import glob
import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

fits_file = '/glade/work/rjarolim/data/SST/campaign_2023_1050/BLOS_panorama1050_6173_StkIQUV_rebin.fits'

data_los = fits.getdata('/glade/work/rjarolim/data/SST/campaign_2023_1050/BLOS_panorama1050_6173_StkIQUV_rebin.fits')
data_trv = fits.getdata('/glade/work/rjarolim/data/SST/campaign_2023_1050/BTRV_panorama1050_6173_StkIQUV_rebin.fits')
header = fits.getheader(fits_file)

data = (data_los - data_los[0, 0]) - (data_trv - data_trv[0, 0])
mask = (data == 0).astype(np.float32)

# save mask to a fits file
hdu = fits.PrimaryHDU(data=mask, header=header)
hdu.writeto('/glade/work/rjarolim/data/SST/campaign_2023_1050/mask.fits', overwrite=True)
# save mask to a png file
plt.imsave('/glade/work/rjarolim/data/SST/campaign_2023_1050/mask.png', mask, cmap='gray')
plt.imsave('/glade/work/rjarolim/data/SST/campaign_2023_1050/data1.png', data, vmin=0, vmax=500)
plt.imsave('/glade/work/rjarolim/data/SST/campaign_2023_1050/data2.png', fits.getdata('/glade/work/rjarolim/data/SST/campaign_2023_1050/BTRV_panorama1050_6173_StkIQUV_rebin.fits'), vmin=0, vmax=500)

converted_path = '/glade/work/rjarolim/data/SST/campaign_2023_1050_converted'
os.makedirs(converted_path, exist_ok=True)
for f in glob.glob('/glade/work/rjarolim/data/SST/campaign_2023_1050/*.fits'):
    d, h  = fits.getdata(f), fits.getheader(f)
    d = np.flip(d, 0)
    ft = os.path.join(converted_path, os.path.basename(f))
    fits.writeto(ft, d, h, overwrite=True)

