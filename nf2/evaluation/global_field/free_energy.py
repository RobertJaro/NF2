import glob
import os

import numpy as np
import pfsspy
import torch
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map
from torch import nn

from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian
from nf2.evaluation.unpack import load_coords

from astropy import units as u

# cr = 2213
for cr in reversed(range(2213, 2268, 5)):
    files = f'/glade/work/rjarolim/nf2/synoptic/series_v3/{cr}.nf2'
    results_path = '/glade/work/rjarolim/nf2/synoptic/series_v3/results'
    synoptic_map_path = f'/glade/work/rjarolim/data/global/synoptic/hmi.synoptic_mr_polfil_720s.{cr}.Mr_polfil.fits'

    # files = f'/glade/work/rjarolim/nf2/global/2173_vp_series_v1/*.nf2'
    # results_path = '/glade/work/rjarolim/nf2/global/2173_vp_series_v1/results'
    # synoptic_map_path = f'/glade/work/rjarolim/data/global/fd_2173/hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits'

    os.makedirs(results_path, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    files = sorted(glob.glob(files))

    for file in files:
        state = torch.load(file, map_location=device)
        model = nn.DataParallel(state['model'])

        # PFSS extrapolation
        potential_r_map = Map(synoptic_map_path)
        potential_r_map = potential_r_map.resample([360, 180] * u.pix)


        spherical_boundary_coords = all_coordinates_from_map(potential_r_map)
        coords = np.stack([np.stack([
            np.ones_like(spherical_boundary_coords.lat.value) * r,
            spherical_boundary_coords.lat.to_value(u.rad) + np.pi / 2,
            spherical_boundary_coords.lon.to_value(u.rad),
            ], -1) for r in np.linspace(1, 1.3, 128)], 0)

        # radius = 1.3
        # coords = np.stack(np.meshgrid(
        #     np.linspace(1, radius, 128),
        #     np.linspace(0, np.pi, 180),
        #     np.linspace(0, 2 * np.pi, 360), indexing='ij'), -1)

        cartesian_coords = spherical_to_cartesian(coords)

        b, j = load_coords(model, 1, state['b_norm'], cartesian_coords, device, progress=True, compute_currents=True)
        b_rtp = vector_spherical_to_cartesian(b, coords)

        # potential_r_map.data[np.isnan(potential_r_map.data)] = 0
        potential_r_map.data[:, :] = b_rtp[0, :, :, 0]
        pfss_in = pfsspy.Input(potential_r_map, 128, 2.5)
        pfss_out = pfsspy.pfss(pfss_in)

        spherical_boundary_coords = SkyCoord(lon=coords[..., 2] * u.rad, lat=(coords[..., 1] - np.pi / 2) * u.rad,
                                             radius=coords[..., 0] * u.solRad, frame=potential_r_map.coordinate_frame)
        potential_shape = spherical_boundary_coords.shape  # required workaround for pfsspy spherical reshape
        spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
        spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
        spherical_boundary_values[..., 1] *= -1  # flip B_theta
        potential_b = np.stack([spherical_boundary_values[..., 0],
                                              spherical_boundary_values[..., 1],
                                              spherical_boundary_values[..., 2]], -1)

        # compute free energy
        free_energy = (b ** 2).sum(-1) - (potential_b ** 2).sum(-1)

        extent = [0, 360, -90, 90]
        # plot free energy
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(free_energy.sum(0), origin='lower',
                       norm=SymLogNorm(1, vmin=-1e5, vmax=1e5),
                       cmap='RdBu_r', extent=extent)
        ax.set_title('Free Energy')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        # grid with steps of 20
        ax.grid()
        ax.set_xticks(np.arange(0, 360, 20))
        ax.set_yticks(np.arange(-80, 90, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(os.path.join(results_path,
                                 f'free_energy_{os.path.basename(file).replace("nf2", "jpg")}'),
                    dpi=300)
        plt.close(fig)

        # free energy v2
        free_energy = np.linalg.norm(b - potential_b, axis=-1)

        extent = [0, 360, -90, 90]
        # plot free energy
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(free_energy.sum(0), origin='lower',
                       norm=LogNorm(vmin=10, vmax=1e4),
                       cmap='cividis', extent=extent)
        ax.set_title('Free Energy')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        # grid with steps of 20
        ax.grid()
        ax.set_xticks(np.arange(0, 360, 20))
        ax.set_yticks(np.arange(-80, 80, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(os.path.join(results_path, f'free_energy_v2_{os.path.basename(file).replace("nf2", "jpg")}'),
                    dpi=300)
        plt.close(fig)

        # plot integrated currents
        fig, ax = plt.subplots(figsize=(10, 5))
        current_density = (j ** 2).sum(-1) ** 0.5
        im = ax.imshow(current_density.sum(0), origin='lower',
                       norm=LogNorm(vmin=1e3, vmax=1e6),
                       cmap='plasma', extent=extent)
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
        im = ax.imshow(b[0, :, :, 0], origin='lower',
                       vmin=-100, vmax=100,
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


