import argparse
import glob
import os
import tarfile
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.unpack import load_cube

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()

series_base_path = args.nf2_path
evaluation_path = os.path.join(series_base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

nf2_paths = sorted(glob.glob(os.path.join(series_base_path, '**', 'extrapolation_result.nf2')))
Mm_per_pix = 360e-3
z_pixels = int(np.ceil(20 / (2 * Mm_per_pix)))  # 20 Mm --> pixels; bin2

# save results as npy files
free_energy_files = []
for path in tqdm(nf2_paths):
  save_path = os.path.join(evaluation_path, '%s.npy' % path.split('/')[-2])
  if os.path.exists(save_path):
    free_energy_files += [save_path]
    continue
  b = load_cube(path, progress=False, z=z_pixels, strides=args.strides)
  free_me = get_free_mag_energy(b, progress=False)
  np.save(save_path, free_me)
  free_energy_files += [save_path]

series_dates = [datetime.strptime(os.path.basename(f)[:13], '%Y%m%d_%H%M%S') for f in free_energy_files]

# plot the energy depletion
me_history = [None] * 4
for f, d in zip(free_energy_files, series_dates):
  free_me = np.load(f)
  prev_me = me_history.pop(0)
  if prev_me is None:
    prev_me = free_me
  fig, axs = plt.subplots(1, 2, figsize=(8, 4))
  axs[0].imshow(free_me.sum(2).transpose(), vmin=0, vmax=1e5, origin='lower', cmap='jet',)
  axs[0].set_title(d.isoformat(' '))
  axs[0].set_axis_off()
  axs[1].imshow(np.clip(free_me - prev_me, 0, None).sum(2).transpose(), vmin=0, vmax=1e4, origin='lower', cmap='jet',)
  axs[1].set_title(d.isoformat(' '))
  axs[1].set_axis_off()
  me_history += [free_me]
  #
  plt.tight_layout()
  plt.savefig(os.path.join(evaluation_path, 'free_energy_%s.jpg' % d.isoformat('T')))
  plt.close()

tar = tarfile.open(os.path.join(evaluation_path, 'free_energy_maps.tar.gz'), "w:gz")
for name in glob.glob(os.path.join(evaluation_path, 'free_energy_*.jpg'), recursive=True):
    tar.add(name, arcname=os.path.basename(name))

tar.close()