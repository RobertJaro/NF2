import argparse
import glob
import os

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map, all_coordinates_from_map

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
    mag_dates = np.array([parse(os.path.basename(f).split('.')[2].replace('_TAI', 'Z').replace('_', 'T')) for f in
                          full_disc_files['Br']])

    os.makedirs(results_path, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    overwrite = False
    img_res = (256, 256)
    sampling = 128
    radius_range = [1.0, 1.3]
    dr = ((radius_range[1] - radius_range[0]) * u.solRad).to(u.cm) / sampling

    alpha_norm = Normalize(vmin=0, vmax=10)
    j_norm = LogNorm(vmin=1e10, vmax=3e12)
    energy_norm = LogNorm(vmin=1e10, vmax=5e13)
    euv_norm = LogNorm(vmin=50, vmax=5e3)

    files = sorted(glob.glob(files))

    for i, file in enumerate(files):
        print(f'Processing {file} ({i + 1}/{len(files)})')
        date = parse(os.path.basename(file).split('.')[0].replace('_TAI', 'Z').replace('_', 'T'))

        img_path = os.path.join(results_path, f'disk_{date.strftime("%Y%m%dT%H%M%S")}.png')

        if os.path.exists(img_path) and not overwrite:
            continue

        model = SphericalOutput(file)

        # EUV
        euv_file = euv_files[np.argmin(np.abs(euv_dates - date))]
        euv_map = Map(euv_file)

        exposure_time = euv_map.exposure_time.to_value(u.s)
        euv_map = euv_map.resample(img_res * u.pix)
        euv_coords = all_coordinates_from_map(euv_map)
        radius = np.sqrt(euv_coords.Tx ** 2 + euv_coords.Ty ** 2) / euv_map.rsun_obs
        alpha_mask = np.clip(1.3 - radius, 0, None) / 0.3
        alpha_mask = alpha_mask ** 3
        alpha_mask[radius < 1.0] = 1

        # mag
        # mag_file = full_disc_files['Br'][np.argmin(np.abs(mag_dates - date))]
        # mag_map = Map(mag_file)
        # mag_map = mag_map.reproject_to(euv_map.wcs)  # transform to smaller (HMI) subframe

        # load coordinates
        coords = all_coordinates_from_map(euv_map).transform_to('heliographic_carrington')
        coords = np.stack([
            np.ones_like(coords.lat.to_value(u.rad)),
            coords.lat.to_value(u.rad),
            coords.lon.to_value(u.rad)], axis=-1)
        coords = coords[None].repeat(sampling, axis=0)
        radius_array = np.linspace(radius_range[0], radius_range[1], sampling)
        coords[..., 0] = radius_array[:, None, None]

        skycoords = SkyCoord(lon=coords[..., 2] * u.rad, lat=coords[..., 1] * u.rad, radius=coords[..., 0] * u.solRad,
                             frame='heliographic_carrington')

        model_out = model.load_spherical_coords(skycoords, metrics=['j', 'alpha'], progress=True,
                                                batch_size=int(2 ** 14))

        br = vector_cartesian_to_spherical(model_out['b'], model_out['spherical_coords'])[0, ..., 0]

        j = np.linalg.norm(model_out['metrics']['j'], axis=-1).sum(0) * dr
        j = j.to_value(u.G * u.cm / u.s)

        ff_energy = energy(model_out['b']).sum(0) * dr
        ff_energy = ff_energy.to_value(u.G ** 2 * u.cm)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        ax = axs[0]
        im = ax.imshow(euv_map.data / exposure_time, norm=euv_norm, cmap=euv_map.cmap, origin='lower', alpha=alpha_mask)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Intensity [DN/s]', orientation='vertical')

        ax = axs[1]
        im = ax.imshow(j, norm=j_norm, cmap='inferno', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Current Density [G cm / s]', orientation='vertical')

        ax = axs[2]
        im = ax.imshow(ff_energy, norm=energy_norm, cmap='jet', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Magnetic Energy [erg / cm^2]', orientation='vertical')

        [ax.set_axis_off() for ax in axs]
        axs[0].set_title(date.strftime('%Y-%m-%d %H:%M:%S'))

        fig.tight_layout()
        fig.savefig(img_path, dpi=300)
        plt.close(fig)

        # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        #
        # ax = axs[0]
        # im = ax.imshow(mag_map.data, cmap='gray', origin='lower', vmin=-500, vmax=500)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.05)
        # plt.colorbar(im, cax=cax, label='$B_\\text{r}$ [G]', orientation='vertical')
        #
        # ax = axs[1]
        # im = ax.imshow(br.to_value(u.G), cmap='gray', origin='lower', vmin=-500, vmax=500)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.05)
        # plt.colorbar(im, cax=cax, label='$B_\\text{r}$ [G]', orientation='vertical')
        #
        # [ax.set_axis_off() for ax in axs]
        # axs[0].set_title(date.strftime('%Y-%m-%d %H:%M:%S'))
        # fig.tight_layout()
        # mag_img_path = os.path.join(results_path, f'mag_{date.strftime("%Y%m%dT%H%M%S")}.png')
        # fig.savefig(mag_img_path, dpi=300)
        # plt.close(fig)
