import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.nddata import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/401_v5'
mag_reference = Map('/gpfs/gpfs0/robert.jarolim/data/nf2/401/hmi.sharp_cea_720s.401.20110309_163600_TAI.Br.fits')

model_path = f'{base_path}/extrapolation_result.nf2'
result_path = f'{base_path}/evaluation'
batch_size = 2048

os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state = torch.load(model_path, map_location=device)
model = nn.DataParallel(state['height_mapping_model'])
cube_shape = state['cube_shape']
spatial_norm = state['spatial_norm']

heights = np.array([0, 2 / 360e-3])
ranges = np.array([0, 100 / 360e-3])

for i, (h, h_r) in enumerate(zip(heights, ranges)):
    coords = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], 0:1], -1).astype(np.float32)
    coords[:, :, :, 2] = h

    coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
    coords_shape = coords.shape
    coords = coords.view((-1, 3))

    cube = []
    it = range(int(np.ceil(coords.shape[0] / batch_size)))
    for k in it:
        coord = coords[k * batch_size: (k + 1) * batch_size]
        coord = coord.to(device)
        # init range
        r = torch.zeros(coord.shape[0], 2)
        r[:, 1] = h_r / spatial_norm
        r = r.to(device)

        cube += [model(coord, r).detach().cpu()]

    cube = torch.cat(cube)
    cube = cube.view(*coords_shape).numpy() * spatial_norm * 360e-3

    fig = plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    v_min = 0
    v_max = h_r * 360e-3
    im = ax.imshow(cube[..., 0, 2].T, origin='lower', vmin=v_min, vmax=v_max, cmap='cividis')
    ax.contour(mag_reference.data, origin='lower', colors=['darkgray', 'white'], levels=[-1000, 1000])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.set_axis_off()
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(label='Height [Mm]', size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'height_{h:.2f}.png'), dpi=300, transparent=True)
    plt.close(fig)

