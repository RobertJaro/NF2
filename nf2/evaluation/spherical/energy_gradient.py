import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

from nf2.data.util import spherical_to_cartesian
from nf2.evaluation.unpack import load_coords

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
                            np.linspace(1, radius, 512),
                            np.linspace(0, 2 * np.pi, 1024),
                            np.deg2rad(270), indexing='ij'), -1)
    coords = coords[:, :, 0]

    cartesian_coords = spherical_to_cartesian(coords)

    b, j = load_coords(model, 1, state['b_norm'], cartesian_coords, device, progress=True, compute_currents=True)

    # create polar plot
    energy = np.linalg.norm(b, axis=-1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    thetas_radians, radii = coords[:, :, 1], coords[:, :, 0]
    im = ax.pcolormesh(thetas_radians, radii, energy, norm=LogNorm(1e1, 1e3), cmap='viridis')
    ax.set_rlim(0, radius)
    ax.set_thetalim(0, np.pi)
    ax.set_theta_zero_location('N')
    plt.colorbar(im, ax=ax, pad=0.1, shrink=0.8, label='Energy density [ergs/cm$^3$]')
    plt.savefig(os.path.join(results_path, f'energy_polar_{os.path.basename(file)}.jpg'), dpi=300)
    plt.close()

    # create current plot
    current_density = np.linalg.norm(j, axis=-1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    thetas_radians, radii = coords[:, :, 1], coords[:, :, 0]
    im = ax.pcolormesh(thetas_radians, radii, current_density, norm=LogNorm(1e3, 1e5), cmap='plasma')
    ax.set_rlim(0, radius)
    ax.set_thetalim(0, np.pi)
    ax.set_theta_zero_location('N')
    plt.colorbar(im, ax=ax, pad=0.1, shrink=0.8, label='Current density [G/cm$^3$]')
    plt.savefig(os.path.join(results_path, f'current_density_polar_{os.path.basename(file)}.jpg'), dpi=300)
    plt.close()

    # create energy gradient plot
    energy_gradient = np.gradient(energy, axis=0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    thetas_radians, radii = coords[:, :, 1], coords[:, :, 0]
    im = ax.pcolormesh(thetas_radians, radii, energy_gradient, norm=SymLogNorm(1, vmin=-10, vmax=10), cmap='RdBu_r')
    ax.set_rlim(0, radius)
    ax.set_thetalim(0, np.pi)
    ax.set_theta_zero_location('N')
    plt.colorbar(im, ax=ax, pad=0.1, shrink=0.8, label='Energy gradient [ergs/cm$^3$]')
    plt.savefig(os.path.join(results_path, f'energy_gradient_polar_{os.path.basename(file)}.jpg'), dpi=300)
    plt.close()
