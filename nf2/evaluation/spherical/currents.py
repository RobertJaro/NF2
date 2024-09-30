import argparse
import glob
import os

import numpy as np
import pfsspy
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map, make_heliographic_header

from nf2.data.util import vector_cartesian_to_spherical
from nf2.evaluation.metric import energy
from nf2.evaluation.output import SphericalOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK files', required=False, default=None)

    args = parser.parse_args()

    files = args.nf2_path
    results_path = args.out_path if args.out_path is not None else os.path.dirname(args.nf2_path)

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
    euv_files = sorted(glob.glob("/glade/work/rjarolim/data/global/fd_2173/euv/*.193.image_lev1.fits"))

    euv_dates = np.array([parse(os.path.basename(f).split('.')[2]) for f in euv_files])
    mag_dates = np.array([parse(os.path.basename(f).split('.')[2].replace('_TAI', 'Z').replace('_', 'T')) for f in full_disc_files['Br']])

    os.makedirs(results_path, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    synoptic_map_br = Map(synoptic_files['Br'])
    synoptic_map_bt = Map(synoptic_files['Bt'])
    synoptic_map_bp = Map(synoptic_files['Bp'])


    def _build_synoptic_map(br_file):
        br = Map(br_file)
        synoptic_map_br.meta['date-obs'] = br.date.to_datetime().date_str()
        br = br.reproject_to(synoptic_map_br.wcs)

        br_data = synoptic_map_br.data
        br_data[~np.isnan(br.data)] = br.data[~np.isnan(br.data)]

        br_map = Map(br_data, synoptic_map_br.meta)
        return br_map


    sampling = [128, 128, 128]
    longitude_range = [150, 210]
    latitude_range = [60, 120]
    radius_range = [1.0, 1.3]
    dr = ((radius_range[1] - radius_range[0]) * u.solRad).to_value(u.cm) / sampling[0]

    alpha_norm = Normalize(vmin=0, vmax=10)
    j_norm = LogNorm(vmin=1e10, vmax=3e12)
    energy_norm = LogNorm(vmin=1e10, vmax=5e13)
    euv_norm = LogNorm(vmin=50, vmax=5e3)

    files = sorted(glob.glob(files, recursive=True))

    for file in files:
        print(file)
        # date = parse(os.path.basename(file).split('.')[0].replace('_TAI', 'Z').replace('_', 'T'))
        ds = file.split(os.sep)[-2].split('_')
        date = parse(f'{ds[0]}T{ds[1]}Z')
        date_str = date.isoformat("T", timespec="minutes").replace('+00:00', '')
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

        # build PFSS model
        pfss_in = pfsspy.Input(br_map, 100, 2.5)
        pfss_out = pfsspy.pfss(pfss_in)

        # EUV
        euv_file = euv_files[np.argmin(np.abs(euv_dates - date))]
        euv_map = Map(euv_file)

        exposure_time = euv_map.exposure_time.to_value(u.s)
        carr_header = make_heliographic_header(euv_map.date, euv_map.observer_coordinate, (180 * 8, 360 * 8),
                                               frame='carrington', projection_code='CAR',
                                               map_center_longitude=180 * u.deg)
        euv_map = euv_map.reproject_to(carr_header)

        # MAG
        mag_file = full_disc_files['Br'][np.argmin(np.abs(mag_dates - date))]
        mag_map = Map(mag_file)
        carr_header = make_heliographic_header(mag_map.date, mag_map.observer_coordinate, (180 * 8, 360 * 8),
                                               frame='carrington', projection_code='CAR',
                                               map_center_longitude=180 * u.deg)
        mag_map = mag_map.reproject_to(carr_header)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(euv_map.data / exposure_time, origin='lower', cmap=euv_map.cmap, extent=[0, 360, 180, 0],
                  norm=euv_norm)
        ax.set_xlim(*longitude_range)
        ax.set_ylim(*reversed(latitude_range))
        fig.savefig(os.path.join(results_path, f'euv_{date_str}.jpg'), dpi=300)
        plt.close(fig)

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

        extent = [*longitude_range, *reversed(latitude_range)]

        # B field
        fig, ax = plt.subplots(figsize=(5, 5))
        b_r = vector_cartesian_to_spherical(b, model_out['spherical_coords'])[0, ..., 0]
        im = ax.imshow(b_r, origin='upper', vmin=-500, vmax=500,
                       cmap='gray', extent=extent)
        ax.set_title('B field')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(os.path.join(results_path, f'b_{date_str}.jpg'), dpi=300)
        plt.close(fig)

        # B observed
        fig, ax = plt.subplots(figsize=(5, 5))
        b_r = mag_map.data
        im = ax.imshow(b_r, origin='lower', vmin=-500, vmax=500,
                       cmap='gray', extent=[0, 360, 180, 0])
        ax.set_title('B observed')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_xlim(*longitude_range)
        ax.set_ylim(*reversed(latitude_range))
        fig.savefig(os.path.join(results_path, f'b_obs_{date_str}.jpg'), dpi=300)
        plt.close(fig)



        # currents
        fig, ax = plt.subplots(figsize=(5, 5))
        current_density = np.linalg.norm(j, axis=-1)
        im = ax.imshow(current_density.sum(0) * dr, origin='upper', norm=j_norm, cmap='inferno', extent=extent)
        ax.set_title('Currents')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(os.path.join(results_path, f'currents_{date_str}.jpg'), dpi=300)
        plt.close(fig)

        alpha[np.linalg.norm(b, axis=-1) < 5] = 0
        # alpha
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(alpha.sum(0) * dr, origin='upper', cmap='viridis', extent=extent, norm=alpha_norm)
        ax.set_title(r'$\alpha$')
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(os.path.join(results_path, f'alpha_{date_str}.jpg'), dpi=300)
        plt.close(fig)


        # free energy
        def _plot_energy(ax, energy, title):
            im = ax.imshow(energy, origin='upper', cmap='jet', extent=extent, norm=energy_norm)
            ax.set_title(title)
            ax.set_xlabel('Longitude [deg]')
            ax.set_ylabel('Latitude [deg]')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical', label='[erg/cm^2]')


        integrated_free_energy = free_energy.sum(0) * dr
        integrated_free_energy[integrated_free_energy < 1e8] = 1e8
        integrated_potential_energy = potential_energy.sum(0) * dr
        integrated_ff_energy = ff_energy.sum(0) * dr

        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_energy(ax, integrated_free_energy, 'Free Energy')
        fig.savefig(os.path.join(results_path, f'free_energy_{date_str}.jpg'), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_energy(ax, integrated_potential_energy, 'Potential Energy')
        fig.savefig(os.path.join(results_path, f'potential_energy_{date_str}.jpg'), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_energy(ax, integrated_ff_energy, 'Force-Free Energy')
        fig.savefig(os.path.join(results_path, f'ff_energy_{date_str}.jpg'), dpi=300)
        plt.close(fig)

        # plot overview video
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        ax = axs[0]
        im = ax.imshow(euv_map.data / exposure_time, origin='lower', cmap=euv_map.cmap, extent=[0, 360, 0, 180],
                       norm=euv_norm)
        ax.set_title('SDO/AIA 193 $\AA$')
        ax.set_xlim(*longitude_range)
        ax.set_ylim(*latitude_range)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='[DN/s]')

        # ax = axs[1]
        # im = ax.imshow(b_r, origin='upper', vmin=-500, vmax=500, cmap='gray', extent=extent)
        # ax.set_title('$B_{z}$')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='2%', pad=0.05)
        # fig.colorbar(im, cax=cax, orientation='vertical', label='[G]')

        ax = axs[1]
        im = ax.imshow(current_density.sum(0) * dr, origin='upper', norm=j_norm, cmap='inferno', extent=extent)
        ax.set_title('Current density')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='[G cm / s]')

        ax = axs[2]
        im = ax.imshow(alpha.sum(0) * dr, origin='upper', cmap='viridis', extent=extent, norm=alpha_norm)
        ax.set_title(r'$\alpha$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='[cm$^{-1}$]')

        ax = axs[3]
        im = ax.imshow(integrated_free_energy, origin='upper', cmap='jet', extent=extent, norm=energy_norm)
        ax.set_title('Free Energy')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='[erg/cm$^2$]')

        axs[0].set_ylabel('Latitude [deg]')
        [ax.set_xlabel('Longitude [deg]') for ax in axs]

        fig.tight_layout()

        fig.savefig(os.path.join(results_path, f'video_{date_str}.jpg'), dpi=300)
        plt.close(fig)
