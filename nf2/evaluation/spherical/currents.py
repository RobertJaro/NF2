import glob
import os

import numpy as np
import pfsspy
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.util import vector_cartesian_to_spherical
from nf2.evaluation.metric import energy
from nf2.evaluation.output import SphericalOutput

files = f'/glade/work/rjarolim/nf2/spherical/2173_full_v07/*.nf2'
results_path = '/glade/work/rjarolim/nf2/spherical/2173_full_v07/results'

synoptic_files = {
    'Br': "/glade/work/rjarolim/data/global/fd_2173/hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits",
    'Bt': "/glade/work/rjarolim/data/global/fd_2173/hmi.b_synoptic.2173.Bt.fits",
    'Bp': "/glade/work/rjarolim/data/global/fd_2173/hmi.b_synoptic.2173.Bp.fits"
}
full_disc_files = {
    'Br': sorted(glob.glob("/glade/work/rjarolim/data/global/fd_2173/full_disk/*Br.fits")),
    'Bt': sorted(glob.glob("/glade/work/rjarolim/data/global/fd_2173/full_disk/*Bt.fits")),
    'Bp': sorted(glob.glob("/glade/work/rjarolim/data/global/fd_2173/full_disk/*Bp.fits"))
}
os.makedirs(results_path, exist_ok=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

synoptic_map_br = Map(synoptic_files['Br'])
synoptic_map_bt = Map(synoptic_files['Bt'])
synoptic_map_bp = Map(synoptic_files['Bp'])


def _build_synoptic_map(br_file):
    br = Map(br_file)
    synoptic_map_br.meta['date-obs'] = br.date.to_datetime().isoformat()
    br = br.reproject_to(synoptic_map_br.wcs)

    br_data = synoptic_map_br.data
    br_data[~np.isnan(br.data)] = br.data[~np.isnan(br.data)]

    br_map = Map(br_data, synoptic_map_br.meta)
    return br_map


sampling = [256, 256, 256]
longitude_range = [140, 220]
latitude_range = [50, 130]
radius_range = [1.0, 1.3]
dr = ((radius_range[1] - radius_range[0]) * u.solRad).to_value(u.cm) / sampling[0]

alpha_norm = Normalize()
j_norm = LogNorm()
energy_norm = LogNorm()

files = sorted(glob.glob(files))

for i, file in enumerate(files):
    model = SphericalOutput(file)
    # prepare synoptic data
    # br_map = _build_synoptic_map(full_disc_files['Br'][i])
    # br_map = br_map.resample([360, 180] * u.pix)

    br_map = synoptic_map_br.resample([360, 180] * u.pix)
    spherical_coords = all_coordinates_from_map(br_map)
    model_out = model.load_spherical_coords(spherical_coords)
    b_r = vector_cartesian_to_spherical(model_out['b'].to_value(u.G),
                                        model_out['spherical_coords'])[..., 0]
    br_map.data[:] = b_r

    pfss_in = pfsspy.Input(br_map, 100, 2.5)
    pfss_out = pfsspy.pfss(pfss_in)

    model_out = model.load_spherical(radius_range * u.solRad,
                                     longitude_range=longitude_range * u.deg, latitude_range=latitude_range * u.deg,
                                     metrics=['j', 'alpha'], sampling=sampling, progress=True)

    spherical_coords = model_out['spherical_coords']
    spherical_coords = SkyCoord(lon=spherical_coords[..., 2] * u.rad,
                                lat=(np.pi / 2 - spherical_coords[..., 1]) * u.rad,
                                radius=spherical_coords[..., 0] * u.solRad,
                                frame=br_map.coordinate_frame)
    pfss_cube_shape = spherical_coords.shape
    pfss_b = pfss_out.get_bvec(spherical_coords.reshape((-1,)))
    pfss_b = pfss_b.reshape(*pfss_cube_shape, 3)
    potential_energy = energy(pfss_b)

    j = model_out['metrics']['j'].to_value(u.G / u.s)
    b = model_out['b'].to_value(u.G)
    alpha = model_out['metrics']['alpha'].to_value(u.cm ** -1)
    ff_energy = energy(b)
    free_energy = ff_energy - potential_energy

    extent = [*longitude_range, *latitude_range]

    # B field
    fig, ax = plt.subplots(figsize=(10, 5))
    b_r = b[0, ..., 0]
    im = ax.imshow(b_r, origin='upper', vmin=-500, vmax=500,
                   cmap='gray', extent=extent)
    ax.set_title('B field')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'b_field_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # currents
    fig, ax = plt.subplots(figsize=(10, 5))
    current_density = np.linalg.norm(j, axis=-1)
    im = ax.imshow(current_density.sum(0) * dr, origin='upper', norm=j_norm, cmap='inferno', extent=extent)
    ax.set_title('Currents')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path, f'currents_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)

    # alpha
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(alpha.sum(0) * dr, origin='upper', cmap='viridis', extent=extent, norm=alpha_norm)
    ax.set_title(r'$\alpha$')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(results_path,
                             f'alpha_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)


    # free energy
    def _plot_energy(ax, energy, title):
        im = ax.imshow(energy, origin='upper', cmap='jet', extent=extent, norm=energy_norm)
        ax.set_title(title)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    integrated_free_energy = free_energy.sum(0) * dr
    integrated_free_energy[integrated_free_energy < 1e8] = 1e8
    integrated_potential_energy = potential_energy.sum(0) * dr
    integrated_ff_energy = ff_energy.sum(0) * dr

    _plot_energy(axs[0], integrated_free_energy, 'Free Energy')
    _plot_energy(axs[1], integrated_potential_energy, 'Potential Energy')
    _plot_energy(axs[2], integrated_ff_energy, 'Force-Free Energy')
    fig.tight_layout()
    plt.savefig(os.path.join(results_path,
                             f'free_energy_{os.path.basename(file).replace("nf2", "jpg")}'),
                dpi=300)
    plt.close(fig)
