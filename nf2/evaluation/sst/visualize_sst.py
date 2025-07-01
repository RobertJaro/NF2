from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sst_data = fits.getdata('/glade/work/rjarolim/data/SST/panorama_8542_StkI.fits')[10]
extent = [0, sst_data.shape[1] * 0.09, 0, sst_data.shape[0] * 0.09]

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.imshow(sst_data, cmap='gray', origin='lower', extent=extent, vmin=0.1e-8, vmax=3e-8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')

fig.tight_layout()
fig.savefig('/glade/work/rjarolim/data/SST/pre_eruption.png', dpi=300, transparent=True)
plt.close()

########################################

sst_data = fits.getdata('/glade/work/rjarolim/data/SST/panorama1050_8542_StkI.fits')[10]
extent = [0, sst_data.shape[1] * 0.09, 0, sst_data.shape[0] * 0.09]

fig, ax = plt.subplots(figsize=(4, 3))

im = ax.imshow(sst_data, cmap='gray', origin='upper', extent=extent, vmin=0.1e-8, vmax=3e-8)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
ax.set_xlabel('X [Mm]')
ax.set_ylabel('Y [Mm]')

fig.tight_layout()
fig.savefig('/glade/work/rjarolim/data/SST/post_eruption.png', dpi=300, transparent=True)
plt.close()

fits.getheader('/glade/work/rjarolim/data/SST/panorama1050_8542_StkI.fits')