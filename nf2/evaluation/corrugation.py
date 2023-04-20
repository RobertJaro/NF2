import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.nddata import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from tqdm import tqdm

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_v1'
data_path = '/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'
model_path = f'{base_path}/extrapolation_result.nf2'
result_path = f'{base_path}/evaluation'
batch_size = 2048

os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dict_data = dict(np.load(data_path))

height_maps = dict_data['z_line'] / (dict_data['dy'] * 2) - 20
average_heights = np.median(height_maps, axis=(1, 2))  # use spatial scaling of horizontal field
max_heights = np.max(height_maps, axis=(1, 2))

b_cube = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_cube = np.moveaxis(b_cube, 0, -2)
b_cube = block_reduce(b_cube, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution

state = torch.load(model_path, map_location=device)
model = nn.DataParallel(state['height_mapping_model'])
cube_shape = state['cube_shape']
spatial_norm = state['spatial_norm']

fig, axs = plt.subplots(len(height_maps), 2, figsize=(8, 4 * len(height_maps)))

for i in range(len(height_maps)):
    height_map = height_maps[i] * (0.192 * 2)
    b = b_cube[:, :, i, 2]
    #
    im = axs[i, 0].imshow(b.T, origin='lower', vmin=-np.abs(b).max(), vmax=np.abs(b).max(), cmap='gray')
    divider = make_axes_locatable(axs[i, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='$B_z$ [Gauss]')
    #
    v_min = 0
    im = axs[i, 1].imshow(height_map.T, origin='lower', vmin=v_min)
    divider = make_axes_locatable(axs[i, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')

[ax.set_axis_off() for ax in np.ravel(axs)]
fig.tight_layout()
fig.savefig(os.path.join(result_path, f'tau.jpg'), dpi=300)
plt.close(fig)

for h, h_max, height_map in zip(average_heights, max_heights, height_maps):
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
        r[:, 1] = h_max / spatial_norm
        r = r.to(device)

        cube += [model(coord, r).detach().cpu()]

    cube = torch.cat(cube)
    cube = cube.view(*coords_shape).numpy() * spatial_norm

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    v_min = 0
    v_max = np.max(height_map) * (0.192 * 2)
    im = axs[0].imshow(cube[..., 0, 2].T * (0.192 * 2), origin='lower', vmin=v_min, vmax=v_max)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')
    im = axs[1].imshow(height_map.T * (0.192 * 2), origin='lower', vmin=v_min, vmax=v_max)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')
    [ax.set_axis_off() for ax in axs]
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'height_{h:02.1f}.jpg'), dpi=300)
    plt.close(fig)