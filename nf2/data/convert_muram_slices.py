import numpy as np
import pandas as pd
from astropy.nddata import block_reduce

input = '/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'
output = '/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/muram_slices.npy'

# prepare data
# Tau slices file
# Bx, By, Bz, Babs: Gauss
# mu (inclination), azimuth: degrees
# dx, dy, dz, z_line: cm
# tau_lev: no units
# x is the vertical direction (64 km/pix)
# y, z are in the horizontal plane (192 km/pix)
# Dimensions: (nb_of_tau_levels, ny, nz)
dict_data = dict(np.load(input))
b_cube = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_cube = np.moveaxis(b_cube, 0, -2)
b_cube = block_reduce(b_cube, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution

# save data cube
np.save(output, b_cube)


height_maps = dict_data['z_line'] / (dict_data['dy'] * 2) # use spatial scaling of horizontal field
height_maps -= 20 # shift 0 to photosphere
height_maps = block_reduce(height_maps, (1, 2, 2), np.mean)  # reduce to HMI resolution
# set first map fixed to photosphere
# height_maps[0, :, :] = 0
# adjust for slices
# height_maps = height_maps[:b_cube.shape[2]]
average_heights = np.median(height_maps, axis=(1, 2))
max_heights = np.max(height_maps, axis=(1, 2))

# print average heights with 3 decimal places
print('Average heights', np.round(average_heights, 3))
# print max heights with 3 decimal places
print('Max heights', np.round(max_heights, 3))

# save height information to txt file with pandas (using three decimal places)
df = pd.DataFrame({'average_heights': average_heights, 'max_heights': max_heights})
df.to_csv(output.replace('.npy', '.txt'), index=False, sep=' ', float_format='%.3f')