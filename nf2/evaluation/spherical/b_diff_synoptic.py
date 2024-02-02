import argparse
import os

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch import nn

from nf2.data.util import spherical_to_cartesian, vector_cartesian_to_spherical
from nf2.evaluation.unpack import load_coords

parser = argparse.ArgumentParser(description='Evaluate NF2 on synoptic data')
parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('--out_path', type=str, help='path to the target VTK files')
parser.add_argument('--cr', type=int, help='Carrington rotation number')

args = parser.parse_args()

nf2_file = args.nf2_path
results_path = args.out_path
carrington_rotation = args.cr
os.makedirs(results_path, exist_ok=True)

ref_r_file = f"/glade/work/rjarolim/data/global/synoptic/hmi.synoptic_mr_polfil_720s.{carrington_rotation}.Mr_polfil.fits",
ref_t_file = f"/glade/work/rjarolim/data/global/synoptic/hmi.b_synoptic.{carrington_rotation}.Bt.fits"
ref_p_file = f"/glade/work/rjarolim/data/global/synoptic/hmi.b_synoptic.{carrington_rotation}.Bp.fits",

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ref_r_map = Map(ref_r_file)
ref_t_map = Map(ref_t_file)
ref_p_map = Map(ref_p_file)
coords = all_coordinates_from_map(ref_r_map).transform_to(frames.HeliographicCarrington)
ref_b = np.stack([ref_r_map.data, -ref_t_map.data, ref_p_map.data]).T * u.G

spherical_coords = np.stack([
    coords.radius.value,
    np.pi / 2 + coords.lat.to(u.rad).value,
    coords.lon.to(u.rad).value,
]).transpose()
cartesian_coords = spherical_to_cartesian(spherical_coords)

# Load the model
state = torch.load(nf2_file, map_location=device)
model = nn.DataParallel(state['model'])

b = load_coords(model, 1, state['data']['G_per_dB'], cartesian_coords, device, progress=True, compute_currents=False)

b = vector_cartesian_to_spherical(b, spherical_coords)

extent = [coords.lon.min().value, coords.lon.max().value, coords.lat.min().value, coords.lat.max().value]
# create difference plot
fig, axs = plt.subplots(3, 3, figsize=(25, 15))
for i in range(3):
    im = axs[i, 0].imshow(ref_b[..., i].value.T, cmap='grey', vmin=-500, vmax=500, origin='lower', extent=extent)
    divider = make_axes_locatable(axs[i, 0])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, label='B$_r$ [G]')

    im = axs[i, 1].imshow(b[..., i].value.T, cmap='grey', vmin=-500, vmax=500, origin='lower', extent=extent)
    divider = make_axes_locatable(axs[i, 1])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, label='B$_{\\theta}$ [G]')

    im = axs[i, 2].imshow(np.abs(b[..., i] - ref_b[..., i]).value.T, cmap='viridis', vmin=0, vmax=100, origin='lower', extent=extent)
    divider = make_axes_locatable(axs[i, 2])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, label='B$_{\\phi}$ [G]')

axs[0, 0].set_title('Reference')
axs[0, 1].set_title('NF2')
axs[0, 2].set_title('Difference')
plt.tight_layout()
plt.savefig(os.path.join(results_path, f'difference.jpg'), dpi=300)
plt.close()

print('Mean absolute difference:', np.nanmean(((b - ref_b) ** 2).sum(-1) ** 0.5))
