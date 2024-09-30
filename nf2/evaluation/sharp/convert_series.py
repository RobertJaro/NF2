import argparse
import glob
import os
import pickle

import numpy as np
import torch.cuda
from astropy import units as u
from tqdm import tqdm

from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy, vector_norm, divergence, theta_J, sigma_J
from nf2.evaluation.output import CartesianOutput


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    # list of nf2 paths
    parser.add_argument('--nf2_dir', type=str, help='path to the directory with the NF2 files', required=True, nargs='+')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--height_range', type=float, help='height range', required=False, nargs=2, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='Mm per pixel', required=False, default=0.72)
    parser.add_argument('--batch_size', type=int, help='batch size', required=False, default=int(2 ** 13))
    args = parser.parse_args()

    nf2_files = [sorted(glob.glob(f)) for f in args.nf2_dir]
    nf2_files = [f for files in nf2_files for f in files] # flatten list
    result_path = args.result_path if args.result_path is not None else args.nf2_path
    os.makedirs(result_path, exist_ok=True)

    # evaluate series
    batch_size = args.batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size
    convert_nf2_series(nf2_files, result_path,
                       height_range=args.height_range,
                       Mm_per_pixel=args.Mm_per_pixel, batch_size=batch_size)


def evaluate_nf2(nf2_file, **kwargs):
    out = CartesianOutput(nf2_file)
    res = out.load_cube(metrics=['j'], **kwargs)

    b = res['b']
    j = res['metrics']['j']
    a = res['a']
    Mm_per_pixel = res['Mm_per_pixel'] * u.Mm

    me = energy(b).value * (u.erg * u.cm ** -3)  # erg = G^2 cm^3
    free_me = get_free_mag_energy(b.to_value(u.G), progress=False) * (u.erg * u.cm ** -3)
    theta = theta_J(b.value, j.value)
    jxb = np.cross(j, b, axis=-1)

    result = {
        'time': out.time,
        'integrated_quantities': {
            'energy': me.sum() * Mm_per_pixel ** 3,
            'free_energy': free_me.sum() * Mm_per_pixel ** 3
        },
        'metrics': {
            'divergence': (np.abs(divergence(b)) / vector_norm(b)).mean(),
            'jxb': vector_norm(jxb).mean(),
            'theta': theta,
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


def convert_nf2_series(nf2_paths, result_path, overwrite=True, **kwargs):
    for nf2_file in tqdm(nf2_paths):
        save_file = os.path.join(result_path, os.path.basename(nf2_file).replace('.nf2', '.pkl'))
        if os.path.exists(save_file) and overwrite is False:
            continue
        result = evaluate_nf2(nf2_file, **kwargs)
        with open(save_file, 'wb') as f:
            pickle.dump(result, f)


def load_results(series_results):
    series_results = series_results if isinstance(series_results, list) else sorted(glob.glob(series_results))
    # load files
    integrated_quantities = {'energy': [], 'free_energy': []}
    metrics = {'divergence': [], 'jxb': [], 'theta': []}
    maps = {'b_0': [], 'current_density_map': [], 'energy_map': [], 'free_energy_map': [], 'jxb_map': []}
    height_distribution = {'height_free_energy': [], 'height_current_density': []}
    times = []
    wcs = []
    Mm_per_pixel = None
    for f in tqdm(series_results, desc='Loading files'):
        with open(f, 'rb') as file:
            data = pickle.load(file)
            integrated_quantities['energy'].append(data['integrated_quantities']['energy'])
            integrated_quantities['free_energy'].append(data['integrated_quantities']['free_energy'])
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
