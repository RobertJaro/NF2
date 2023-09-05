import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.nddata import block_reduce
from matplotlib.colors import PowerNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from tqdm import tqdm

base_path = '/gpfs/gpfs0/robert.jarolim/multi_height/muram_2tau_Bxyz_split'
data_path = '/gpfs/gpfs0/robert.jarolim/data/nf2/multi_height/tau_slices_B_extrapolation.npz'
model_path = f'{base_path}/extrapolation_result.nf2'
result_path = f'{base_path}/evaluation'
batch_size = 2048

os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dict_data = dict(np.load(data_path))

b_cube = np.stack([dict_data['By'], dict_data['Bz'], dict_data['Bx']], -1) * np.sqrt(4 * np.pi)
b_cube = np.moveaxis(b_cube, 0, -2)
b_cube = block_reduce(b_cube, (2, 2, 1, 1), np.mean)  # reduce to HMI resolution


height_maps = dict_data['z_line'] / (dict_data['dy'] * 2) - 20


height_maps = height_maps[[0, -3, -2]]
b_cube = b_cube[:, :, [0, -3, -2]]

height_maps = block_reduce(height_maps, (1, 2, 2), np.mean)
average_heights = np.median(height_maps, axis=(1, 2))  # use spatial scaling of horizontal field
max_heights = np.max(height_maps, axis=(1, 2))

print('Average heights', average_heights)
print('Max heights', max_heights)

print('Fixed heights avg distance:', np.mean(np.abs(average_heights[:, None, None] - height_maps), (1, 2)) * (0.192 * 2),
      np.mean(np.abs(average_heights[:, None, None] - height_maps), (1, 2)) / height_maps.max((1,2)) * 100 )



state = torch.load(model_path, map_location=device)
model = nn.DataParallel(state['height_mapping_model'])
cube_shape = state['cube_shape']
spatial_norm = state['spatial_norm']

height_mapping = state['height_mapping']

fig, axs = plt.subplots(len(height_maps), 4, figsize=(14, 3 * len(height_maps)))

height_diffs = []
height_diffs_relative = []
for i, (h, h_min, h_max, height_map) in enumerate(zip(height_mapping['z'], height_mapping['z_min'], height_mapping['z_max'], height_maps)):
    coords = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], 0:1], -1).astype(np.float32)
    coords[:, :, :, 2] = h
    #
    coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
    coords_shape = coords.shape
    coords = coords.view((-1, 3))
    #
    cube = []
    it = range(int(np.ceil(coords.shape[0] / batch_size)))
    for k in it:
        coord = coords[k * batch_size: (k + 1) * batch_size]
        coord = coord.to(device)
        # init range
        r = torch.zeros(coord.shape[0], 2)
        r[:, 0] = h_min / spatial_norm
        r[:, 1] = h_max / spatial_norm
        r = r.to(device)
        #
        cube += [model(coord, r).detach().cpu()]
    #
    cube = torch.cat(cube)
    cube = cube.view(*coords_shape).numpy() * spatial_norm
    #
    b = b_cube[:, :, i, 2]
    #
    extent = [0, cube_shape[0] * (0.192 * 2), 0, cube_shape[1] * (0.192 * 2)]
    #
    axs[i, 0].set_title(fr'$\tau = 10^{{-{i + 1}}}$', fontsize=20)
    im = axs[i, 0].imshow(b.T, origin='lower', vmin=-1500, vmax=1500, cmap='gray', extent=extent)
    axs[i, 0].set_xlabel('x [Mm]', fontsize=12)
    axs[i, 0].set_ylabel('y [Mm]', fontsize=12)
    divider = make_axes_locatable(axs[i, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical').set_label(label='$B_z$ [Gauss]', size=15, rotation=270, labelpad=15)
    #
    v_min = 0
    v_max = np.max(height_map) * (0.192 * 2)
    norm = PowerNorm(0.5, vmin=v_min, vmax=v_max)
    #
    im = axs[i, 1].imshow(height_map.T * (0.192 * 2), origin='lower', norm=norm, extent=extent)
    axs[i, 1].set_xlabel('x [Mm]', fontsize=12)
    axs[i, 1].set_ylabel('y [Mm]', fontsize=12)
    divider = make_axes_locatable(axs[i, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical').set_label(label='Height [Mm]', size=15, rotation=270, labelpad=15)
    #
    im = axs[i, 2].imshow(cube[..., 0, 2].T * (0.192 * 2), origin='lower', norm=norm, extent=extent)
    axs[i, 2].set_xlabel('x [Mm]', fontsize=12)
    axs[i, 2].set_ylabel('y [Mm]', fontsize=12)
    divider = make_axes_locatable(axs[i, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical').set_label(label='Height [Mm]', size=15, rotation=270, labelpad=15)
    # add 2D histogram
    _, _, _, im = axs[i, 3].hist2d(np.clip(height_map.flatten(), 0, None) * (0.192 * 2), cube[..., 0, 2].flatten() * (0.192 * 2), bins=100, norm=LogNorm(), cmap='magma')
    axs[i, 3].plot([v_min, v_max], [v_min, v_max], 'k--', c='red')
    axs[i, 3].set_xlim(v_min, v_max)
    axs[i, 3].set_ylim(v_min, v_max)
    axs[i, 3].set_xlabel(r'$z_{\rm ref}$ [Mm]', size=12)
    axs[i, 3].set_ylabel(r"$z'$ [Mm]", size=12)
    # set aspect ratio square
    axs[i, 3].set_aspect('equal')
    divider = make_axes_locatable(axs[i, 3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical').set_label(label='[Counts]', size=15, rotation=270, labelpad=15)
    #
    height_diffs += [np.abs(height_map - cube[..., 0, 2]).mean() * (0.192 * 2)]
    height_diffs_relative += [np.abs(height_map - cube[..., 0, 2]).mean() / height_map.max() * 100]

axs[-1, 3].set_xlim(0, 12)
axs[-1, 3].set_ylim(0, 12)

fig.tight_layout()
fig.savefig(os.path.join(result_path, f'tau.jpg'), dpi=300)
plt.close(fig)

pd.DataFrame({'height': average_heights, 'height_diff': height_diffs, 'height_diff_relative': height_diffs_relative}).to_csv(
    os.path.join(result_path, 'height_diffs.csv'), index=False)
