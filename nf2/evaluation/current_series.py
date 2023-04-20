import glob
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from nf2.evaluation.unpack import load_cube
from nf2.train.metric import curl, vector_norm

result_path = '/gpfs/gpfs0/robert.jarolim/nf2/6975/j_maps'
model_paths = sorted(
    glob.glob('/gpfs/gpfs0/robert.jarolim/nf2/6975/series/**/extrapolation_result.nf2', recursive=True))

os.makedirs(result_path, exist_ok=True)

for i, model_path in tqdm(enumerate(model_paths), total=len(model_paths)):
    b = load_cube(model_path, progress=False, z=128, strides=2)
    j = curl(b)
    # plot integrated current density
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.imshow(vector_norm(j).sum(2).transpose(), origin='lower', cmap='inferno', vmin=0, vmax=2e3)
    plt.title('$|J|$')
    print('Max |J|:', vector_norm(j).sum(2).transpose().max())
    plt.axis('off')
    plt.subplot(212)
    jxb_h = np.cross(j, b, -1)[..., :2]
    jxb_h = (vector_norm(jxb_h) / vector_norm(b[..., :2])).sum(2)
    plt.imshow(jxb_h.transpose(), origin='lower', cmap='viridis', norm=LogNorm(vmin=1e1, vmax=1.5e3, clip=True))
    plt.title('$|J x B|$')
    print('Max |JxB|/|B|:', jxb_h.max())
    plt.axis('off')
    plt.savefig(os.path.join(result_path, 'j_%04d.jpg' % i))
    plt.close(fig)

shutil.make_archive(result_path, 'zip', result_path)

