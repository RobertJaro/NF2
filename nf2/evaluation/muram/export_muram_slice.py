import numpy as np
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt

from nf2.loader.muram import MURaMSnapshot
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

Mm_per_pixel = 0.192  # 0.192
mfr_muram_snapshot = MURaMSnapshot('/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D', iteration=474000)


bx = mfr_muram_snapshot.Bx[:, :, :]
by = mfr_muram_snapshot.By[:, :, :]
bz = mfr_muram_snapshot.Bz[:, :, :]
b = np.stack([bx, by, bz], axis=-1) * np.sqrt(4 * np.pi)

tau = mfr_muram_snapshot.tau[:, :, :]
temp = mfr_muram_snapshot.T[:, :, :]
rho = mfr_muram_snapshot.rho[:, :, :]
vx = mfr_muram_snapshot.vx[:, :, :]
vy = mfr_muram_snapshot.vy[:, :, :]
vz = mfr_muram_snapshot.vz[:, :, :]
v = np.stack([vx, vy, vz], axis=-1)


# integer division
resolution = 0.192 * u.Mm / u.pix
assert resolution % mfr_muram_snapshot.ds[0] == 0, f'resolution {resolution} must be a multiple of {mfr_muram_snapshot.ds[0]}'
assert resolution % mfr_muram_snapshot.ds[1] == 0, f'resolution {resolution} must be a multiple of {mfr_muram_snapshot.ds[1]}'
assert resolution % mfr_muram_snapshot.ds[2] == 0, f'resolution {resolution} must be a multiple of {mfr_muram_snapshot.ds[2]}'
x_binning = resolution // mfr_muram_snapshot.ds[0]
y_binning = resolution // mfr_muram_snapshot.ds[1]
z_binning = resolution // mfr_muram_snapshot.ds[2]

b = block_reduce(b, (x_binning, y_binning, z_binning, 1), np.mean)
tau = block_reduce(tau, (x_binning, y_binning, z_binning), np.mean)
temp = block_reduce(temp, (x_binning, y_binning, z_binning), np.mean)
rho = block_reduce(rho, (x_binning, y_binning, z_binning), np.mean)
v = block_reduce(v, (x_binning, y_binning, z_binning, 1), np.mean)


x_slice = 200
extent = [0, b.shape[1] * Mm_per_pixel, 0, b.shape[2] * Mm_per_pixel]

# create a plot with |B|, T, rho, |v|
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

ax = axs[0]
im = ax.imshow(np.linalg.norm(b[x_slice], axis=-1).T, origin='lower', cmap='viridis', norm='log', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, label='|B| [G]')

ax = axs[1]
im = ax.imshow(temp[x_slice].T, origin='lower', cmap='inferno', norm='log', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, label='T [K]')

ax = axs[2]
im = ax.imshow(rho[x_slice].T, origin='lower', cmap='plasma', norm='log', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, label='rho [g/cm^3]')

ax = axs[3]
im = ax.imshow(np.linalg.norm(v[x_slice], axis=-1).T, origin='lower', cmap='cividis', norm='log', extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax, label='|v| [cm/s]')

for ax in axs:
    ax.set_xlabel('Y [Mm]')
    ax.set_ylabel('Z [Mm]')

fig.tight_layout()
fig.show()

plt.savefig('/glade/work/rjarolim/data/muram/slice.png')
plt.close()

out_dict = {'B[G]': b, 'tau': tau, 'T[K]': temp, 'rho[g/cm^3]': rho, 'v[cm/s]': v,
            'Mm_per_pixel': Mm_per_pixel}
np.savez('/glade/work/rjarolim/data/muram/cube.npz', **out_dict)

# export slice
out_dict = {'B[G]': b[x_slice], 'tau': tau[x_slice], 'T[K]': temp[x_slice], 'rho[g/cm^3]': rho[x_slice], 'v[cm/s]': v[x_slice],
            'Mm_per_pixel': Mm_per_pixel}
np.savez('/glade/work/rjarolim/data/muram/slice.npz', **out_dict)