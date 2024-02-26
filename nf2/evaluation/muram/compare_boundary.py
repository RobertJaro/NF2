import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.output import HeightTransformOutput
from nf2.loader.muram import MURaMSnapshot
import argparse

from astropy import units as u

parser = argparse.ArgumentParser(description='Evaluate MURaM snapshot.')
parser.add_argument('--nf2_path', type=str, help='path to the nf2 project.')
parser.add_argument('--source_path', type=str, help='path to the MURaM simulation.')
parser.add_argument('--iteration', type=int, help='iteration of the snapshot.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

nf2_out = HeightTransformOutput(args.nf2_path)

nf2_heights = list(nf2_out.load_height_mapping())

snapshot = MURaMSnapshot(args.source_path, args.iteration)
tau_cube = snapshot.tau

pix_height = np.argmin(np.abs(tau_cube - 1), axis=2) * u.pix
Mm_height = (pix_height * snapshot.ds[2]).to(u.Mm)
base_height_Mm = Mm_height.min()
base_height_pix = pix_height.min()
cube_height = (snapshot.shape[2] * u.pix * snapshot.ds[2]).to(u.Mm)
print(f'Cube Height: {cube_height} [Mm]; {snapshot.shape[2]} [pix]')
print(f'Base Height: {base_height_Mm} [Mm]; {base_height_pix} [pix]')

fig, axs = plt.subplots(2, 4, figsize=(16, 4))

for i, (tau, nf2_height) in enumerate(zip([1.0, 0.001, 0.000100, 0.000001], nf2_heights)):
    row = axs[:, i]
    #
    pix_height = np.argmin(np.abs(tau_cube - tau), axis=2) * u.pix - base_height_pix
    Mm_height = (pix_height * snapshot.ds[2]).to(u.Mm)
    #
    nf2_z = nf2_height['coords'][:, :, 0, 2]
    #
    vmax = max(Mm_height.max(), nf2_z.max()).value
    #
    ax = row[0]
    im = ax.imshow(Mm_height.value.T, vmin=0, vmax=vmax, cmap='inferno', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Height [Mm]')
    ax.set_title(f'Tau: {tau:.1e}')
    #
    ax = row[1]
    im = ax.imshow(nf2_z.value.T, vmin=0, vmax=vmax, cmap='inferno', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Height [Mm]')
    ax.set_title(f'NF2 Height {i + 1:1d}')

fig.tight_layout()
fig.savefig(os.path.join(args.output, f'compare_boundary.jpg'))
plt.close(fig)