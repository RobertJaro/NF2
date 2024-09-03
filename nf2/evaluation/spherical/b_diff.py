import os

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.util import spherical_to_cartesian, vector_cartesian_to_spherical
from nf2.evaluation.output import SphericalOutput

nf2_file = '/glade/work/rjarolim/nf2/spherical/2173_free_v06/extrapolation_result.nf2'
results_path = '/glade/work/rjarolim/nf2/spherical/2173_free_v06/results'
ref_r_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Br.fits'
ref_t_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Bt.fits'
ref_p_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Bp.fits'
ref_r_err_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Br_err.fits'
ref_t_err_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Bt_err.fits'
ref_p_err_file = '/glade/work/rjarolim/data/global/fd_2173/full_disk/hmi.b_720s.20160205_170000_TAI.Bp_err.fits'
os.makedirs(results_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ref_r_map = Map(ref_r_file)
ref_t_map = Map(ref_t_file)
ref_p_map = Map(ref_p_file)

ref_r_err_map = Map(ref_r_err_file)
ref_t_err_map = Map(ref_t_err_file)
ref_p_err_map = Map(ref_p_err_file)

err_b = np.stack([ref_r_err_map.data, ref_t_err_map.data, ref_p_err_map.data]).T

coords = all_coordinates_from_map(ref_r_map).transform_to(frames.HeliographicCarrington)
ref_b = np.stack([ref_r_map.data, ref_t_map.data, ref_p_map.data]).T

spherical_coords = np.stack([
    coords.radius.to(u.solRad).value,
    np.pi / 2 - coords.lat.to(u.rad).value,
    coords.lon.to(u.rad).value,
]).transpose()
cartesian_coords = spherical_to_cartesian(spherical_coords)

# Load the model
model = SphericalOutput(nf2_file)
model_out = model.load_coords(cartesian_coords)

b_xyz = model_out['b']
b = vector_cartesian_to_spherical(b_xyz, spherical_coords).to_value(u.G)

# create difference plot
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    im = axs[i, 0].imshow(ref_b[..., i].T, cmap='grey', vmin=-500, vmax=500)
    plt.colorbar(im, ax=axs[i, 0], pad=0.1, shrink=0.8, label='B [G]')
    axs[i, 0].set_title('Reference')
    im = axs[i, 1].imshow(b[..., i].T, cmap='grey', vmin=-500, vmax=500)
    plt.colorbar(im, ax=axs[i, 1], pad=0.1, shrink=0.8, label='B [G]')
    axs[i, 1].set_title('NF2')
    im = axs[i, 2].imshow(np.clip(np.abs(b[..., i] - ref_b[..., i]) - err_b[..., i], a_min=0, a_max=None).T, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(im, ax=axs[i, 2], pad=0.1, shrink=0.8, label='Difference [G]')
    axs[i, 2].set_title('Difference')
plt.tight_layout()
plt.savefig(os.path.join(results_path, f'difference.jpg'), dpi=300)
plt.close()

b_diff = np.clip(np.abs(b - ref_b) - err_b, a_min=0, a_max=None)
print('Mean absolute difference:', np.nanmean(np.linalg.norm(b_diff, axis=-1)))
