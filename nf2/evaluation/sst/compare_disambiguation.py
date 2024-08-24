import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map

from nf2.data.util import los_trv_azi_to_img, img_to_los_trv_azi
from nf2.evaluation.output import HeightTransformOutput

parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
parser.add_argument('--nf2_path', type=str, help='path to the nf2 project.')
parser.add_argument('--b_los', type=str, help='path to the FITS file containing the line-of-sight magnetic field.')
parser.add_argument('--b_trv', type=str, help='path to the FITS file containing the transverse magnetic field.')
parser.add_argument('--b_azi', type=str, help='path to the FITS file containing the azimuthal magnetic field.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

nf2_out = HeightTransformOutput(args.nf2_path)

b_nf2 = nf2_out.load_cube(height_range=(0, 1))['B']

b_los = Map(args.b_los).data
b_trv = Map(args.b_trv).data
b_azi = Map(args.b_azi).data

b = np.stack([b_los, b_trv, np.pi - b_azi], axis=-1)
b_xyz = los_trv_azi_to_img(b)

print(np.nansum(b - img_to_los_trv_azi(b_xyz)))

fig, axs = plt.subplots(2, 3, figsize=(16, 4))

ax = axs[0, 0]
im = ax.imshow(b_nf2[:, :, 0, 0].value.T, vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Bx [G]')
ax.set_title('NF2')
#
ax = axs[0, 1]
im = ax.imshow(b_nf2[:, :, 0, 1].value.T, vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='By [G]')

ax = axs[0, 2]
im = ax.imshow(b_nf2[:, :, 0, 2].value.T, vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Bz [G]')

ax = axs[1, 0]
im = ax.imshow(b_xyz[:, :, 0], vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Bx [G]')
ax.set_title('SHARP')

ax = axs[1, 1]
im = ax.imshow(b_xyz[:, :, 1], vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='By [G]')

ax = axs[1, 2]
im = ax.imshow(b_xyz[:, :, 2], vmin=-1000, vmax=1000, cmap='gray', origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Bz [G]')

fig.tight_layout()
fig.savefig(os.path.join(args.output, f'compare_disambiguation.jpg'))
plt.close(fig)
