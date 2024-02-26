import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.loader.muram import MURaMSnapshot
import argparse

from astropy import units as u

parser = argparse.ArgumentParser(description='Evaluate MURaM snapshot.')
parser.add_argument('--source_path', type=str, help='path to the MURaM simulation.')
parser.add_argument('--iteration', type=int, help='iteration of the snapshot.')
args = parser.parse_args()


snapshot = MURaMSnapshot(args.source_path, args.iteration)
tau_cube = snapshot.tau

pix_height = np.argmin(np.abs(tau_cube - 1), axis=2) * u.pix
Mm_height = (pix_height * snapshot.ds[2]).to(u.Mm)
base_height_Mm = Mm_height.min()
base_height_pix = pix_height.min()
cube_height = (snapshot.shape[2] * u.pix * snapshot.ds[2]).to(u.Mm)
print(f'Cube Height: {cube_height} [Mm]; {snapshot.shape[2]} [pix]')
print(f'Base Height: {base_height_Mm} [Mm]; {base_height_pix} [pix]')

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for ax, tau in zip(axs, [1.0, 0.001, 0.000100, 0.000001]):
    pix_height = np.argmin(np.abs(tau_cube - tau), axis=2) * u.pix - base_height_pix
    Mm_height = (pix_height * snapshot.ds[2]).to(u.Mm)
    #
    im = ax.imshow(Mm_height.value.T, vmin=0, cmap='inferno', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Height [Mm]')
    ax.set_title(f'Tau: {tau:.1e}')
    #
    print(f'Tau: {tau}')
    print(f'Height: {Mm_height.min()} - {Mm_height.max()}; {Mm_height.mean()}')
    print(f'Height: {pix_height.min()} - {pix_height.max()}; {pix_height.mean()}')

fig.tight_layout()
fig.savefig(f'/glade/work/rjarolim/nf2/multi_height/tau_height.jpg')
plt.close(fig)