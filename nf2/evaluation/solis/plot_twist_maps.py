import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import readsav

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/solis_evaluation'
os.makedirs(base_path, exist_ok=True)

chromospheric = readsav('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/q3d_chromospheric_f01.sav')
photospheric = readsav('/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/q3d_photospheric_f01.sav')


data = chromospheric
l = 'chromospheric'

for i in range(150, 210, 10):
    # plot slice of data cube
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #
    extent = np.array([*data['xreg'], *data['zreg']]) * 0.36
    #
    im = axs[0].imshow(np.log(data['q3d'][:, i - data['yreg'][0], :]), vmin=6, vmax=15, cmap='Greens', origin='lower', extent=extent)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", ticklocation = 'top').set_label('log(Q)', size=18)
    cax.tick_params(labelsize=14)
    axs[0].spines['bottom'].set_color('darkorange')
    axs[0].spines['bottom'].set_linewidth(3)
    axs[0].spines['bottom'].set_linestyle('--')
    axs[0].set_xlabel('X [Mm]', size=18)
    axs[0].set_ylabel('Z [Mm]', size=18)
    #
    im = axs[1].imshow(data['twist3d'][:, i - data['yreg'][0], :], vmin=-2, vmax=2, cmap='bwr', origin='lower', extent=extent)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", ticklocation = 'top').set_label(r"$T_{w}$", size=18)
    cax.tick_params(labelsize=14)
    axs[1].spines['bottom'].set_color('darkorange')
    axs[1].spines['bottom'].set_linewidth(3)
    axs[1].spines['bottom'].set_linestyle('--')
    axs[1].set_xlabel('X [Mm]', size=18)
    #
    fig.tight_layout()
    fig.savefig(os.path.join(base_path, f'{l}_{i:03d}.png'), transparent=True, dpi=300)
    plt.close()

plt.imshow(data['twist3d'][50, :, :], vmin=-2, vmax=2, cmap='bwr', origin='lower')
plt.savefig(os.path.join(base_path, 'test.png'), transparent=True, dpi=300)
plt.close()