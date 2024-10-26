import argparse
import glob
import os
from datetime import timedelta
from multiprocessing import Pool

import drms
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from dateutil.parser import parse
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map

from nf2.data.download import download_euv
from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy
from nf2.evaluation.output import CartesianOutput


class _F:

    def __init__(self, ref_wcs):
        self.ref_wcs = ref_wcs

    def func(self, file):
        s_map = Map(file)
        if s_map.meta['QUALITY'] != 0:
            return None
        exposure = s_map.exposure_time.to_value(u.s)
        reprojected_map = s_map.reproject_to(self.ref_wcs)
        return Map(reprojected_map.data / exposure, self.ref_wcs)


def get_integrated_euv_map(euv_files, ref_wcs):
    with Pool(os.cpu_count()) as p:
        reprojected_maps = p.map(_F(ref_wcs).func, euv_files)
    integrated_euv = np.array([m.data for m in reprojected_maps if m is not None]).sum(0)
    time = (Map(euv_files[-1]).date.datetime - Map(euv_files[0]).date.datetime).total_seconds()
    euv_map = Map(integrated_euv * time, ref_wcs)
    return euv_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the directory of the NF2 files')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--email', type=str, help='email for the DRMS client', required=True)
    parser.add_argument('--date_range', nargs=3, type=str, required=True)
    args = parser.parse_args()

    nf2_files = sorted(glob.glob(args.nf2_path))
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    dates = [parse(os.path.basename(nf2_file).split('.')[0][:-4].replace('_', 'T')) for nf2_file in nf2_files]
    dates = np.array(dates)

    start_time = parse(args.date_range[0])
    end_time = parse(args.date_range[1])
    peak_time = parse(args.date_range[2])

    client = drms.Client(email=args.email, verbose=True)
    euv_files = download_euv(start_time=start_time, end_time=peak_time,
                             dir=result_path, client=client, channel=94)

    filter_dates = (dates > (start_time - timedelta(minutes=12))) & \
                   (dates < (peak_time + timedelta(minutes=12)))
    # filter_dates = (dates > start_time) & (dates < end_time)
    flare_nf2_files = np.array(nf2_files)[filter_dates]

    model_1 = CartesianOutput(flare_nf2_files[0])
    out_1 = model_1.load_cube(Mm_per_pixel=0.72, progress=True)
    free_energy_1 = get_free_mag_energy(out_1['b'].value, progress=False) * (u.erg / u.cm ** 3)
    energy_1 = energy(out_1['b'].value) * (u.erg / u.cm ** 3)

    model_2 = CartesianOutput(flare_nf2_files[-1])
    out_2 = model_2.load_cube(Mm_per_pixel=0.72, progress=True)
    free_energy_2 = get_free_mag_energy(out_2['b'].value, progress=False) * (u.erg / u.cm ** 3)
    energy_2 = energy(out_2['b'].value) * (u.erg / u.cm ** 3)

    Mm_per_pix = out_1['Mm_per_pixel'] * u.Mm

    # released energy
    released_energy = -np.clip(energy_2 - energy_1, a_min=None, a_max=0)
    released_energy_map = (released_energy.sum(2) * Mm_per_pix).to_value(u.erg / u.cm ** 2)
    released_energy_map[released_energy_map < 1e10] = np.nan
    total_released_energy = (released_energy.sum() * Mm_per_pix ** 3).to_value(u.erg)

    # gained energy
    gained_energy = np.clip(energy_2 - energy_1, a_min=0, a_max=None)
    gained_energy_map = (gained_energy.sum(2) * Mm_per_pix).to_value(u.erg / u.cm ** 2)
    gained_energy_map[gained_energy_map < 1e10] = np.nan
    total_gained_energy = (gained_energy.sum() * Mm_per_pix ** 3).to_value(u.erg)

    energy_diff = (energy_2 - energy_1).sum() * Mm_per_pix ** 3
    energy_diff = energy_diff.to_value(u.erg)
    energy_diff_map = (energy_2 - energy_1).sum(2) * Mm_per_pix
    energy_diff_map = energy_diff_map.to_value(u.erg / u.cm ** 2)

    free_energy_diff = (free_energy_2 - free_energy_1).sum() * Mm_per_pix ** 3
    free_energy_diff = free_energy_diff.to_value(u.erg)

    with open(os.path.join(args.result_path, f'{start_time.isoformat("T", timespec="minutes")}.txt'), 'w') as f:
        f.write(f'Total released energy: {total_released_energy * 1e-31:.02f} 10^31 erg\n')
        f.write(f'Total gained energy: {total_gained_energy * 1e-31:.02f} 10^31 erg\n')
        f.write(f'Total energy difference: {energy_diff * 1e-31:.02f} 10^31 erg\n')
        f.write(f'Total free energy difference: {free_energy_diff * 1e-31:.02f} 10^31 erg\n')

    euv_map = get_integrated_euv_map(list(euv_files), model_1.wcs[0])
    [os.remove(f) for f in euv_files]

    extent = (0, released_energy_map.shape[0] * Mm_per_pix.to_value(u.Mm), 0,
              released_energy_map.shape[1] * Mm_per_pix.to_value(u.Mm))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    ax = axs[0]
    im = ax.imshow(euv_map.data, origin='lower', cmap='sdoaia94', norm=LogNorm(np.percentile(euv_map.data, 10)),
                   extent=extent)
    # ax.set_title('Integrated EUV SDO/AIA 94 $\AA$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, label='AIA 94 $\AA$ [DN]')
    ax.contour(released_energy_map.T, levels=[1e12], colors='red', linewidths=1, extent=extent)

    ax = axs[1]
    free_energy_map = (free_energy_1.sum(2) * Mm_per_pix).to_value(u.erg / u.cm ** 2)
    free_energy_map[free_energy_map < 1e11] = 1e11
    im = ax.imshow(free_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=LogNorm(1e11, 5e13))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, label='E$_{free}$ [erg / cm$^2$]')

    ax = axs[2]
    im = ax.imshow(energy_diff_map.T, origin='lower', cmap='seismic', extent=extent,
                   norm=Normalize(vmin=-1e13, vmax=1e13))
    # norm=SymLogNorm(1e12, vmin=-1e14, vmax=1e14))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, label=r'$\Delta \text{E}$ [erg / cm$^2$]')

    [ax.set_xlim(100, 400) for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(args.result_path, f'fl_{start_time.isoformat("T", timespec="minutes")}'), dpi=300,
                transparent=True)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    ax = axs[0]
    im = ax.imshow(gained_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=LogNorm(1e10, 1e13))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, label=r'$\Delta \text{E}_\text{+}$ [erg / cm$^2$]')
    ax.set_title('Gained Energy')

    ax = axs[1]
    im = ax.imshow(released_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=LogNorm(1e10, 1e13))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax, label=r'$\Delta \text{E}_\text{-}$ [erg / cm$^2$]')
    ax.set_title('Released Energy')

    [ax.set_xlim(100, 400) for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(args.result_path, f'energy_{start_time.isoformat("T", timespec="minutes")}'), dpi=300,
                transparent=True)
    plt.close()
