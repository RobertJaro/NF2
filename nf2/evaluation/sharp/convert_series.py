import argparse
import glob
import os
import pickle
from copy import copy

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize, LogNorm
from matplotlib.dates import date2num, DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm

from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy, vector_norm, divergence, theta_J
from nf2.evaluation.output import CartesianOutput


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the directory of the NF2 files', required=True)
    parser.add_argument('--euv_path', type=str, help='path to the directory of the EUV files', required=True)
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--max_height', type=float, help='max height of the volume in Mm', required=False, default=20)
    parser.add_argument('--Mm_per_pixel', type=float, help='Mm per pixel', required=False, default=0.72)
    parser.add_argument('--add_flares', action='store_true', help='query for flares and plot', required=False,
                        default=None)
    args = parser.parse_args()

    nf2_files = sorted(glob.glob(args.nf2_path))
    result_path = args.result_path if args.result_path is not None else args.nf2_path
    os.makedirs(result_path, exist_ok=True)

    # evaluate series
    series_results = convert_nf2_series(nf2_files, result_path,
                                        height_range=[0, args.max_height],
                                        Mm_per_pixel=args.Mm_per_pixel, batch_size=int(2 ** 14))

    times, integrated_quantities, metrics, maps, height_distribution, wcs = load_results(series_results)

    # plot integrated quantities
    _plot_integrated_qunatities(times, integrated_quantities, height_distribution,
                                result_path,
                                args.Mm_per_pixel, add_flares=args.add_flares)

    _plot_map_videos(times, maps, wcs, args.euv_path, result_path, args.Mm_per_pixel)


def evaluate_nf2(nf2_file, **kwargs):
    out = CartesianOutput(nf2_file)
    res = out.load_cube(**kwargs)

    b = res['b']
    j = res['j']
    a = res['a']
    Mm_per_pixel = res['Mm_per_pixel'] * u.Mm

    me = energy(b).value * (u.erg * u.cm ** -3)  # erg = G^2 cm^3
    free_me = get_free_mag_energy(b.to_value(u.G), progress=False) * (u.erg * u.cm ** -3)
    theta = theta_J(b.value, j.value)
    magnetic_helicity = (a * b).sum(-1)
    current_helicity = (b * j).sum(-1)
    jxb = np.cross(j, b, axis=-1)

    result = {
        'time': out.time,
        'integrated_quantities': {
            'energy': me.sum() * Mm_per_pixel ** 3,
            'free_energy': free_me.sum() * Mm_per_pixel ** 3,
            'current_helicity': current_helicity.sum() * Mm_per_pixel ** 3,
            'magnetic_helicity': magnetic_helicity.sum() * Mm_per_pixel ** 3,
        },
        'metrics': {
            'divergence': (np.abs(divergence(b)) / vector_norm(b)).mean(),
            'jxb': vector_norm(jxb).mean(),
            'theta': theta
        },
        'maps': {
            'b_0': b[:, :, 0, 2],  # bottom boundary
            'current_density_map': vector_norm(j).sum(2) * Mm_per_pixel,
            'energy_map': me.sum(2) * Mm_per_pixel,
            'free_energy_map': free_me.sum(2) * Mm_per_pixel,
            'jxb_map': vector_norm(jxb).sum(2) * Mm_per_pixel,
        },
        'height_distribution': {
            'height_free_energy': free_me.mean((0, 1)) * Mm_per_pixel ** 2,
            'height_current_density': vector_norm(j).mean((0, 1)) * Mm_per_pixel ** 2,
        },
        'info': {
            'data_config': out.data_config,
            'Mm_per_pixel': res['Mm_per_pixel'],
            'wcs': out.wcs
        }
    }
    return result


def convert_nf2_series(nf2_paths, result_path, **kwargs):
    series_result_path = os.path.join(result_path, 'pkl')
    os.makedirs(series_result_path, exist_ok=True)
    results = []
    for nf2_file in tqdm(nf2_paths):
        save_file = os.path.join(series_result_path, os.path.basename(nf2_file).replace('.nf2', '.pkl'))
        if os.path.exists(save_file):
            results.append(save_file)
            continue
        result = evaluate_nf2(nf2_file, **kwargs)
        with open(save_file, 'wb') as f:
            pickle.dump(result, f)
        results.append(save_file)

    return results


def _plot_map_videos(times, maps, wcs, euv_path, result_path, Mm_per_pixel):
    euv_files = np.array(sorted(glob.glob(euv_path)))
    euv_dates = np.array([parse(fits.getheader(f, 1)['DATE-OBS']) for f in euv_files])

    video_path = os.path.join(result_path, 'video')
    os.makedirs(video_path, exist_ok=True)

    j_norm = LogNorm()
    free_energy_norm = LogNorm(1e9, 1e14)

    for i, time in enumerate(times):
        print(time)
        euv_file = euv_files[np.argmin(np.abs(euv_dates - time))]
        euv_map = Map(euv_file)
        exposure = euv_map.exposure_time.to_value(u.s)
        euv_map = euv_map.reproject_to(wcs[i][0])

        b_0 = maps['b_0'][i].value
        current_density_map = maps['current_density_map'][i].to_value(u.G * u.cm * u.s ** -1)
        free_energy_map = maps['free_energy_map'][i].to_value(u.erg * u.cm ** -2)

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))

        ax = axs[0, 0]

        extent = np.array([0, b_0.shape[0], 0, b_0.shape[1]]) * Mm_per_pixel
        im = ax.imshow(b_0.T, origin='lower', cmap='gray', extent=extent, norm=Normalize(vmin=-2000, vmax=2000))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Magnetic Field - B$_z$')

        ax = axs[0, 1]
        cm = copy(plt.get_cmap('sdoaia131'))
        cm.set_bad('black')
        im = ax.imshow(euv_map.data / exposure, origin='lower', cmap=cm, extent=extent, norm=LogNorm(vmin=5, vmax=1e3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[DN / s]')
        ax.set_title('EUV - AIA 131')

        ax = axs[1, 0]

        im = ax.imshow(current_density_map.T, origin='lower', cmap='inferno', extent=extent, norm=j_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[G cm / s]')
        ax.set_title('Current Density - |J|')

        ax = axs[1, 1]
        im = ax.imshow(free_energy_map.T, origin='lower', cmap='jet', extent=extent, norm=free_energy_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='[erg / cm$^2$]')
        ax.set_title('Free Magnetic Energy')

        [ax.set_xlabel('X [Mm]') for ax in axs[1, :]]
        [ax.set_ylabel('Y [Mm]') for ax in axs[:, 0]]

        fig.suptitle(f'Time: {time.isoformat(" ", timespec="minutes")}')

        plt.tight_layout()
        plt.savefig(os.path.join(video_path, f'{time.isoformat("T", timespec="minutes")}.jpg'), dpi=300)
        plt.close()


def load_results(series_results):
    series_results = series_results if isinstance(series_results, list) else sorted(glob.glob(series_results))
    # load files
    integrated_quantities = {'energy': [], 'free_energy': [], 'current_helicity': [], 'magnetic_helicity': []}
    metrics = {'divergence': [], 'jxb': [], 'theta': []}
    maps = {'b_0': [], 'current_density_map': [], 'energy_map': [], 'free_energy_map': [], 'jxb_map': []}
    height_distribution = {'height_free_energy': [], 'height_current_density': []}
    times = []
    wcs = []
    Mm_per_pixel = None
    for f in series_results:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            integrated_quantities['energy'].append(data['integrated_quantities']['energy'])
            integrated_quantities['free_energy'].append(data['integrated_quantities']['free_energy'])
            integrated_quantities['current_helicity'].append(data['integrated_quantities']['current_helicity'])
            integrated_quantities['magnetic_helicity'].append(data['integrated_quantities']['magnetic_helicity'])
            metrics['divergence'].append(data['metrics']['divergence'])
            metrics['jxb'].append(data['metrics']['jxb'])
            metrics['theta'].append(data['metrics']['theta'])
            maps['b_0'].append(data['maps']['b_0'])
            maps['current_density_map'].append(data['maps']['current_density_map'])
            maps['energy_map'].append(data['maps']['energy_map'])
            maps['free_energy_map'].append(data['maps']['free_energy_map'])
            maps['jxb_map'].append(data['maps']['jxb_map'])
            height_distribution['height_free_energy'].append(data['height_distribution']['height_free_energy'])
            height_distribution['height_current_density'].append(data['height_distribution']['height_current_density'])
            times.append(data['time'])
            wcs.append(data['info']['wcs'])
            Mm_per_pixel = data['info']['Mm_per_pixel']

    integrated_quantities = {k: np.stack(v) for k, v in integrated_quantities.items()}
    metrics = {k: np.stack(v) for k, v in metrics.items()}
    maps = {k: np.stack(v) for k, v in maps.items()}
    height_distribution = {k: np.stack(v) for k, v in height_distribution.items()}
    times = np.array(times)
    wcs = np.array(wcs)

    return {'times': times, 'integrated_quantities': integrated_quantities, 'metrics': metrics, 'maps': maps,
            'height_distribution': height_distribution, 'wcs': wcs, 'Mm_per_pixel': Mm_per_pixel}


if __name__ == '__main__':
    main()
