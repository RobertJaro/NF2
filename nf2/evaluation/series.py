import argparse
import glob
import os
import pickle

import imageio
import numpy as np
import pandas as pd
import torch.cuda
from dateutil.parser import parse
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from matplotlib.dates import date2num, DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm

from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy, curl, vector_norm, divergence, weighted_theta
from nf2.evaluation.unpack import load_cube


def evaluate_nf2(nf2_file, z, cm_per_pixel, strides, batch_size):
    b = load_cube(nf2_file, progress=False, z=z, strides=strides, batch_size=batch_size)
    j = curl(b)
    jxb = np.cross(j, b, axis=-1)
    me = energy(b)
    free_me = get_free_mag_energy(b, progress=False)
    theta = weighted_theta(b, j)

    # check free magnetic energy is positive
    assert free_me.sum() > 0, f'free magnetic energy is negative: {free_me.sum()}'
    date_str = nf2_file.split('/')[-2].split('_')
    result = {
        'date': parse(date_str[0] + 'T' + date_str[1]),
        # torch.load(nf2_file)['meta_data']['DATE-OBS'],
        # bottom boundary
        'b_0': b[:, :, 0, 2],
        # integrated quantities
        'total_energy': me.sum() * cm_per_pixel ** 3,
        'total_free_energy': free_me.sum() * cm_per_pixel ** 3,
        'total_div': (np.abs(divergence(b)) / vector_norm(b)).mean(),
        'total_jxb': vector_norm(jxb).mean(),
        'theta': theta,
        # heigth distribution
        'height_free_energy': free_me.mean((0, 1)),
        # maps
        'j_map': vector_norm(j).sum(2),
        'energy_map': me.sum(2) * cm_per_pixel,
        'free_energy_map': free_me.sum(2) * cm_per_pixel,
        'jxb_map': vector_norm(jxb).sum(2),
    }
    return result


def evaluate_nf2_series(nf2_paths, z, cm_per_pixel, strides, batch_size):
    results = []
    for nf2_file in tqdm(nf2_paths):
        results.append(evaluate_nf2(nf2_file, z, cm_per_pixel, strides, batch_size))
    # concatenate dicts to dict of lists
    results = {k: [r[k] for r in results] for k in results[0].keys()}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)
    parser.add_argument('--add_flares', action='store_true', help='query for flares and plot', required=False,
                        default=None)
    args = parser.parse_args()

    nf2_files = sorted(glob.glob(os.path.join(args.nf2_path, '**', '*.nf2')))
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    batch_size = int(1e5 * torch.cuda.device_count())

    # load simulation scaling
    Mm_per_pix = torch.load(nf2_files[0])['Mm_per_pix'] if 'Mm_per_pix' in torch.load(nf2_files[0]) else 0.36
    z_pixels = int(np.ceil(20 / Mm_per_pix))  # 20 Mm --> pixels

    # adjust scale to strides
    Mm_per_pix *= args.strides
    cm_per_pix = (Mm_per_pix * 1e8)

    # evaluate series
    series_results = evaluate_nf2_series(nf2_files, z_pixels, cm_per_pix, args.strides, batch_size)

    # save with pickle
    with open(os.path.join(result_path, 'results.pkl'), 'wb') as f:
        pickle.dump(series_results, f)

    # restore from pickle
    with open(os.path.join(result_path, 'results.pkl'), 'rb') as f:
        series_results = pickle.load(f)

    # save results as csv
    df = pd.DataFrame({'date': series_results['date'],
                       'energy [ergs]' : series_results['total_energy'], 'free_energy [ergs]' : series_results['total_free_energy'],
                       'div': series_results['total_div'], 'jxb': series_results['total_jxb'], 'theta': series_results['theta']})
    df.to_csv(os.path.join(result_path, 'series.csv'))

    # plot integrated quantities
    x_dates = date2num(series_results['date'])
    date_format = DateFormatter('%d-%H:%M')

    fig, full_axs = plt.subplots(5, 2, figsize=(8, 12), gridspec_kw={"width_ratios": [1, 0.05]})
    axs = full_axs[:, 0]
    [ax.axis('off') for ax in full_axs[:, 1]]
    # make date axis
    for ax in axs:
        ax.xaxis_date()
        ax.set_xlim(x_dates[0], x_dates[-1])
        ax.xaxis.set_major_formatter(date_format)

    fig.autofmt_xdate()

    ax = axs[0]
    ax.plot(x_dates, np.array(series_results['total_energy']) * 1e-32)
    ax.set_ylabel('Energy\n[$10^{32}$ erg]')

    ax = axs[1]
    ax.plot(x_dates, np.array(series_results['total_free_energy']) * 1e-32)
    ax.set_ylabel('Free Energy\n[$10^{32}$ erg]')

    ax = axs[2]
    dt = np.diff(x_dates)[0] / 2
    dz = Mm_per_pix / 2
    free_energy_distribution = np.array(series_results['height_free_energy']).T
    max_height = free_energy_distribution.shape[0] * Mm_per_pix
    mpb = ax.imshow(free_energy_distribution,  # average
                    extent=(x_dates[0] - dt, x_dates[-1] + dt, -dz, max_height + dz), aspect='auto', origin='lower',
                    cmap=cm.get_cmap('jet'), vmin=0)
    ax.set_ylabel('Altitude\n[Mm]')
    ax.set_ylim(0, 20)
    # add colorbar
    full_axs[2, 1].axis('on')
    cbar = fig.colorbar(mpb, cax=full_axs[2, 1], label='Free Energy Density\n' + r'[erg $cm^{-3}]$')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_ticks([2e2, 4e2, 6e2, 8e2])
    full_axs[2, 1].remove()

    ax = axs[3]
    ax.plot(x_dates, np.array(series_results['total_div']))
    ax.set_ylabel(r'$|\mathbf{\nabla} \cdot B| / |B|$' + '\n[1/pixel]')

    ax = axs[4]
    ax.plot(x_dates, np.array(series_results['theta']))
    ax.set_ylabel(r'$\theta_J$' + '\n[deg]')

    if args.add_flares:
        flares = Fido.search(a.Time(min(series_results['date']), max(series_results['date'])),
                             a.hek.EventType("FL"),
                             a.hek.OBS.Observatory == "GOES")["hek"]
        goes_mapping = {c: 10 ** (i) for i, c in enumerate(['B', 'C', 'M', 'X'])}
        for t, goes_class in zip(flares['event_peaktime'], flares['fl_goescls']):
            flare_intensity = np.log10(float(goes_class[1:]) * goes_mapping[goes_class[0]])
            if flare_intensity >= np.log10(1 * 10 ** 2):
                [ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='black') for ax in axs]
            if flare_intensity >= np.log10(1 * 10 ** 3):
                [ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='red') for ax in axs]

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'series.jpg'))
    plt.close()

    # create video of maps
    # j_map
    j_maps = np.array(series_results['j_map'])
    images = [cm.get_cmap('viridis')(Normalize(vmin=0, vmax=j_maps.max())(j_map.T)) for j_map in j_maps]
    images = np.flip(images, axis=1)
    imageio.mimsave(os.path.join(result_path, 'j_maps.mp4'), images)
    # free_energy_map
    free_energy_maps = np.array(series_results['free_energy_map'])
    images = [cm.get_cmap('jet')(Normalize(vmin=0, vmax=free_energy_maps.max())(free_energy_map.T)) for free_energy_map
              in free_energy_maps]
    images = np.flip(images, axis=1)
    imageio.mimsave(os.path.join(result_path, 'free_energy_maps.mp4'), images)
    # free_energy_change_map
    free_energy_change_maps = np.gradient(np.array(series_results['free_energy_map']), axis=0)
    v_min_max = np.max(np.abs(free_energy_change_maps))
    images = [cm.get_cmap('seismic')(Normalize(vmin=-v_min_max, vmax=v_min_max)(free_energy_change_map.T)) for
              free_energy_change_map in free_energy_change_maps]
    images = np.flip(images, axis=1)
    imageio.mimsave(os.path.join(result_path, 'free_energy_change_maps.mp4'), images)
    # jxb_map
    jxb_maps = np.array(series_results['jxb_map'])
    images = [cm.get_cmap('plasma')(Normalize(vmin=0, vmax=jxb_maps.max())(jxb_map.T)) for jxb_map in jxb_maps]
    images = np.flip(images, axis=1)
    imageio.mimsave(os.path.join(result_path, 'jxb_maps.mp4'), images)

    # create video of free energy with timeline
    video_path = os.path.join(result_path, 'free_energy_video')
    os.makedirs(video_path, exist_ok=True)
    for date, free_energy_map in zip(series_results['date'], series_results['free_energy_map']):
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        ax = axs[0]
        im = ax.imshow(free_energy_map.T, vmin=0, vmax=free_energy_maps.max(),
                       origin='lower', cmap='jet',
                       extent=np.array([0, free_energy_map.shape[0], 0, free_energy_map.shape[1]]) * Mm_per_pix)
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        ax.set_title(date)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='Free Energy Density\n' + r'[erg $cm^{-2}]$')
        #
        ax = axs[1]
        ax.plot(x_dates, np.array(series_results['total_free_energy']) * 1e-32)
        ax.set_ylabel('Free Energy\n[$10^{32}$ erg]')
        ax.xaxis_date()
        ax.set_xlim(x_dates[0], x_dates[-1])
        ax.xaxis.set_major_formatter(date_format)
        # tilt x-axis labels
        fig.autofmt_xdate()
        # add flares
        if args.add_flares:
            for t, goes_class in zip(flares['event_peaktime'], flares['fl_goescls']):
                flare_intensity = np.log10(float(goes_class[1:]) * goes_mapping[goes_class[0]])
                if flare_intensity >= np.log10(1 * 10 ** 2):
                    ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='black')
                if flare_intensity >= np.log10(1 * 10 ** 3):
                    ax.axvline(x=date2num(t.datetime), linestyle='dotted', c='red')
        # add vertical line
        ax.axvline(x=date2num(date), linestyle='solid', c='red')
        #
        plt.tight_layout()
        plt.savefig(os.path.join(video_path, 'free_energy_map_' + str(date) + '.jpg'))
        plt.close()

    images = [plt.imread(image) for image in sorted(glob.glob(os.path.join(video_path, '*.jpg')))]
    imageio.mimsave(os.path.join(result_path, 'free_energy_series.mp4'), images)



