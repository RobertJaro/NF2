import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from sunpy.map import Map

synoptic_map = Map('/glade/work/rjarolim/data/global/fd_2173/hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits')
br_map = Map('/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_000000_TAI.Br.fits')

img = Normalize(vmin=-500, vmax=500)(br_map.data)
img = cm.get_cmap('gray')(img)
plt.imsave('/glade/work/rjarolim/nf2/spherical/br.png', img)

synoptic_map.meta['date-obs'] = br_map.date.to_datetime().isoformat()
reprojected = br_map.reproject_to(synoptic_map.wcs)
synoptic_map.data[~np.isnan(reprojected.data)] = np.nan

img = Normalize(vmin=-500, vmax=500)(synoptic_map.data)

img = cm.get_cmap('gray')(img)
plt.imsave('/glade/work/rjarolim/nf2/spherical/synoptic.png', img)
