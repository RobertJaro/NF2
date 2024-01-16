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

from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian
from nf2.evaluation.unpack import load_coords

from astropy import units as u

# cr = 2213

files = f'/glade/work/rjarolim/nf2/global/2173_vp_series_v1/*.nf2'
results_path = '/glade/work/rjarolim/nf2/global/2173_vp_series_v1/results'
synoptic_map_path = f'/glade/work/rjarolim/data/global/fd_2173/hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits'

os.makedirs(results_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

files = sorted(glob.glob(files))[::5]

for file in files:
    state = torch.load(file, map_location=device)
    model = nn.DataParallel(state['model'])

    radius = 1.3
    coords = np.stack(np.meshgrid(
                            np.linspace(1, radius, 128),
                            np.linspace(0, np.pi, 180),
                            np.linspace(0, 2 * np.pi, 360), indexing='ij'), -1)

    cartesian_coords = spherical_to_cartesian(coords)

    b, j = load_coords(model, 1, state['b_norm'], cartesian_coords, device, progress=True, compute_currents=True)

    extent = [0, 360, -90, 90]

    # currents
    fig, ax = plt.subplots(figsize=(10, 5))
    current_density = (j ** 2).sum(-1) ** 0.5
    im = ax.imshow(current_density.sum(0), origin='lower',
                   norm=LogNorm(vmin=1e3, vmax=1e6),
                   cmap='inferno',
                   extent=extent)
    ax.set_title('Currents')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.grid()
    ax.set_xticks(np.arange(0, 360, 20))
    ax.set_yticks(np.arange(-80, 80, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'currents_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # plot photospheric magnetic field
    fig, ax = plt.subplots(figsize=(10, 5))
    b_rtp = vector_spherical_to_cartesian(b, coords)
    im = ax.imshow(b[0, :, :, 0], origin='lower',
                   vmin=-500, vmax=500,
                   cmap='gray', extent=extent)
    ax.set_title('Photospheric Magnetic Field')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.grid()
    ax.set_xticks(np.arange(0, 360, 20))
    ax.set_yticks(np.arange(-80, 80, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'b_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # plot integrated magnetic field
    fig, ax = plt.subplots(figsize=(10, 5))
    energy = (b ** 2).sum(-1) ** 0.5
    im = ax.imshow(energy.sum(0), origin='lower',
                   norm=LogNorm(vmin=1e1, vmax=1e4),
                   cmap='viridis', extent=extent)
    ax.set_title('Integrated Magnetic Field')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.grid()
    ax.set_xticks(np.arange(0, 360, 20))
    ax.set_yticks(np.arange(-80, 80, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'energy_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # vorticity
    fig, ax = plt.subplots(figsize=(10, 5))
    vorticity = np.linalg.norm(j, axis=-1) / np.linalg.norm(b, axis=-1)
    vorticity[np.linalg.norm(b, axis=-1) < 1] = 0
    im = ax.imshow(vorticity.sum(0), origin='lower',
                   vmin=0, vmax=8000,
                   cmap='inferno', extent=extent)
    ax.set_title('Vorticity')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.grid()
    ax.set_xticks(np.arange(0, 360, 20))
    ax.set_yticks(np.arange(-80, 80, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path,
                             f'vorticity_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # plot lorentz force
    fig, ax = plt.subplots(figsize=(10, 5))
    lorentz = np.linalg.norm(np.cross(j, b), axis=-1)
    im = ax.imshow(lorentz.sum(0), origin='lower',
                   norm=LogNorm(vmin=1e3, vmax=1e7),
                   cmap='inferno', extent=extent)
    ax.set_title('Lorentz Force')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.grid()
    ax.set_xticks(np.arange(0, 360, 20))
    ax.set_yticks(np.arange(-80, 80, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'lorentz_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)



