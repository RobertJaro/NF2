import glob
import os

import numpy as np
import pfsspy
import torch
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from torch import nn

from nf2.data.util import spherical_to_cartesian
from nf2.evaluation.unpack import load_coords

from astropy import units as u

files = '/glade/work/rjarolim/global/2154_vp_v4'
results_path = '/glade/work/rjarolim/global/2154_vp_v4/results'
os.makedirs(results_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

files = sorted(glob.glob(os.path.join(files, '*.nf2')))

for file in files:
    state = torch.load(file, map_location=device)
    model = nn.DataParallel(state['model'])

    radius = 1.3
    coords = np.stack(np.meshgrid(
                            np.linspace(1, radius, 10),
                            np.linspace(0, np.pi, 64),
                            np.linspace(0, np.pi, 128), indexing='ij'), -1)
    coords = coords[:, :, 0]

    cartesian_coords = spherical_to_cartesian(coords)

    b, j = load_coords(model, 1, state['b_norm'], cartesian_coords, device, progress=True, compute_currents=True)

    # PFSS extrapolation
    potential_r_map = Map('/glade/work/rjarolim/data/global/fd_2154/hmi.b_synoptic.2154.Br.fits')
    potential_r_map = potential_r_map.resample([360, 180] * u.pix)
    potential_r_map.data[np.isnan(potential_r_map.data)] = 0
    pfss_in = pfsspy.Input(potential_r_map, 100, 2.5)
    pfss_out = pfsspy.pfss(pfss_in)

    spherical_boundary_coords = SkyCoord(lon=coords[..., 2], lat=coords[..., 1], radius=coords[..., 0] * u.solRad,
                                         frame=potential_r_map.frame)
    potential_shape = spherical_boundary_coords.shape  # required workaround for pfsspy spherical reshape
    spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
    spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
    spherical_boundary_values[..., 1] *= -1  # flip B_theta
    potential_b = np.stack([spherical_boundary_values[..., 0],
                                          spherical_boundary_values[..., 1],
                                          spherical_boundary_values[..., 2]]).T

    # compute free energy
    free_energy = (b ** 2).sum(-1) - (potential_b ** 2).sum(-1)

    # plot free energy
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(free_energy, origin='lower', norm=SymLogNorm(1e-5), cmap='RdBu_r')
    ax.set_title('Free Energy')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.linspace(0, 127, 5))
    ax.set_xticklabels(np.linspace(0, 360, 5))
    ax.set_yticks(np.linspace(0, 63, 5))
    ax.set_yticklabels(np.linspace(-90, 90, 5))
    ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, os.path.basename(file) + '_free_energy.jpg'), dpi=300)
    plt.close(fig)

