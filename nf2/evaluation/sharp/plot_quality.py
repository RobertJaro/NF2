import argparse
import glob
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.sharp.convert_series import load_results

from astropy import units as u



def _plot_integrated_qunatities(times, integrated_quantities, height_distribution, result_path, Mm_per_pixel):
    pass


# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
parser.add_argument('--fits_path', type=str, help='path to the directory with the fits files.', required=False, default=None)
parser.add_argument('--nf2_dir', type=str, help='path to the directory with the NF2 files', required=True, nargs='+')
parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
args = parser.parse_args()

result_path = args.result_path

b_r_files = sorted(glob.glob(os.path.join(args.fits_path, '*Br.fits')))
b_t_files = sorted(glob.glob(os.path.join(args.fits_path, '*Bt.fits')))
b_p_files = sorted(glob.glob(os.path.join(args.fits_path, '*Bp.fits')))
b_r_err_files = sorted(glob.glob(os.path.join(args.fits_path, '*Br_err.fits')))
b_t_err_files = sorted(glob.glob(os.path.join(args.fits_path, '*Bt_err.fits')))
b_p_err_files = sorted(glob.glob(os.path.join(args.fits_path, '*Bp_err.fits')))

def _load_map(d):
    br, bt, bp = d['b_r'], d['b_t'], d['b_p']
    br_err, bt_err, bp_err = d['b_r_err'], d['b_t_err'], d['b_p_err']

    b = np.stack([Map(bp).data, -Map(bt).data, Map(br).data]).T
    b_err = np.stack([Map(bp_err).data, Map(bt_err).data, Map(br_err).data]).T

    return {'b': b, 'b_err': b_err}

d = [{'b_r': b_r_file, 'b_t': b_t_file, 'b_p': b_p_file, 'b_r_err': b_r_err_file, 'b_t_err': b_t_err_file, 'b_p_err': b_p_err_file}
     for b_r_file, b_t_file, b_p_file, b_r_err_file, b_t_err_file, b_p_err_file in
     zip(b_r_files, b_t_files, b_p_files, b_r_err_files, b_t_err_files, b_p_err_files)]

with Pool(processes=8) as p:
    res = [r for r in tqdm(p.imap(_load_map, d), total=len(d))]
    b_l = [r['b'] for r in res]
    b_err_l = [r['b_err'] for r in res]

def _load_map_diff(b, b_err, nf2_file):
    out = CartesianOutput(nf2_file)
    res = out.load_slice(batch_size=2**15, compute_jacobian=False, metrics=[])

    b_nf2 = res['b'].to_value(u.G)

    b_diff = np.abs(b - b_nf2)
    b_diff_err = np.clip(b_diff - b_err, a_min=0, a_max=None)

    b_diff = np.linalg.norm(b_diff, axis=-1).mean()
    b_diff_err = np.linalg.norm(b_diff_err, axis=-1).mean()

    # log_image(b, b_nf2, nf2_file)

    return {'b_diff': b_diff, 'b_diff_err': b_diff_err}


def log_image(b, b_nf2, nf2_file):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    ax = axs[0, 0]
    ax.imshow(b[..., 0].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax.set_title('Original')
    ax = axs[0, 1]
    ax.imshow(b_nf2[..., 0].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax.set_title('NF2')
    ax = axs[0, 2]
    ax.imshow(np.abs(b[..., 0] - b_nf2[..., 0]).T, origin='lower', vmin=0, vmax=1000)
    ax.set_title('Difference')
    ax = axs[1, 0]
    ax.imshow(b[..., 1].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax = axs[1, 1]
    ax.imshow(b_nf2[..., 1].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax = axs[1, 2]
    ax.imshow(np.abs(b[..., 1] - b_nf2[..., 1]).T, origin='lower', vmin=0, vmax=1000)
    ax = axs[2, 0]
    ax.imshow(b[..., 2].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax = axs[2, 1]
    ax.imshow(b_nf2[..., 2].T, origin='lower', cmap='gray', vmin=-1000, vmax=1000)
    ax = axs[2, 2]
    ax.imshow(np.abs(b[..., 0] - b_nf2[..., 0]).T, origin='lower', vmin=0, vmax=1000)
    plt.tight_layout()
    fig.savefig(os.path.join(result_path, os.path.basename(nf2_file) + '.jpg'), dpi=300)
    plt.close('all')
    print('diff', np.linalg.norm(b - b_nf2, axis=-1).mean())


nf2_files = [sorted(glob.glob(d)) for d in args.nf2_dir]
nf2_files = [f for d in nf2_files for f in d]
assert len(b_l) == len(b_err_l) == len(nf2_files)
b_diff = []
b_diff_err = []
for b, b_err, nf2_file in tqdm(zip(b_l, b_err_l, nf2_files), total=len(b_l)):
    res = _load_map_diff(b, b_err, nf2_file)
    b_diff.append(res['b_diff'])
    b_diff_err.append(res['b_diff_err'])

output = load_results(args.pkl_path)

times = output['times']
metrics = output['metrics']
height_distribution = output['height_distribution']

Mm_per_pixel = output['Mm_per_pixel']

date_format = DateFormatter('%d-%H:%M')

fig, axs = plt.subplots(3, 1, figsize=(5, 5))

# make date axis
for ax in axs:
    ax.xaxis_date()
    ax.set_xlim(times[0], times[-1])
    ax.xaxis.set_major_formatter(date_format)

fig.autofmt_xdate()

ax = axs[0]
ax.plot(times, b_diff, label=r'$L_{\text{B0}}$')
ax.plot(times, b_diff_err, label=r'$L_{\text{B0, err}}$')
ax.set_ylabel('$\Delta$B [G]')
ax.legend()

ax = axs[1]
ax.plot(times, metrics['divergence'] * 1e4 / Mm_per_pixel)
ax.set_ylabel(r'$\langle |\nabla \cdot \mathbf{B}$| / |B| $\rangle$' + '\n[$10^{-4}$ Mm$^{-1}$]')

ax = axs[2]
ax.plot(times, metrics['theta'])
ax.set_ylabel('$\\theta_{\\text{J}}$ \n[deg]')
ax.set_xlabel('Time')

ax = axs[2].twinx()
ax.plot(times, np.sin(np.deg2rad(metrics['theta'])))
ax.set_ylabel(r'$\sigma_{\text{J}}$')


[ax.axvline(x=datetime(2024, 5, 7, 0, 0), linestyle='dotted', c='black') for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(result_path, 'quality.jpg'), dpi=300)
plt.close()
