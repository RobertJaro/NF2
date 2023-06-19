import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.unpack import load_height_surface

parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
parser.add_argument('base_path', type=str)
args = parser.parse_args()

base_path = args.base_path

model_path = f'{base_path}/extrapolation_result.nf2'
result_path = f'{base_path}/evaluation'
batch_size = 2048

os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state = torch.load(model_path, map_location=device)

height_mapping = state['height_mapping']

# create reference tau
tau = np.stack(np.meshgrid(np.linspace(-1, 1, 64, dtype=np.float32),
                           np.linspace(-1, 1, 64, dtype=np.float32),
                           np.ones(len(height_mapping['z']), dtype=np.float32),
                           indexing='ij'), -1)

for i, c in enumerate(height_mapping['z_max']):
    tau[:, :, i, 2] = c / 64
    sx = sy = c / 64 + 0.5
    x, y = tau[:, :, i, 0], tau[:, :, i, 1]
    gaussian = np.exp(-(x ** 2. / (2. * sx ** 2.) + y ** 2. / (2. * sy ** 2.)))
    gaussian /= gaussian.max()  # normalize
    tau[:, :, i, 2] *= gaussian

tau *= 64

height_surfaces = load_height_surface(model_path)

height_diffs = []
for i, (z, z_min, z_max) in enumerate(zip(height_mapping['z'], height_mapping['z_min'], height_mapping['z_max'])):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    v_min = z_min
    v_max = z_max
    im = axs[0].imshow(tau[..., i, 2].T, origin='lower', vmin=v_min, vmax=v_max)
    axs[0].set_title('Analytic')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [pixels]')
    im = axs[1].imshow(height_surfaces[:, :, i, 2].T, origin='lower', vmin=v_min, vmax=v_max)
    axs[1].set_title('PINN')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    [ax.set_axis_off() for ax in axs]
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [pixels]')
    fig.savefig(os.path.join(result_path, f'height_{z:.2f}.jpg'), dpi=300)
    plt.close(fig)

    z_diff = np.abs(tau[..., i, 2] - height_surfaces[:, :, i, 2]).mean()
    height_diffs += [(z_diff, z_diff / z_max * 100)]

# save height diffs as csv using pandas with two decimal places
df = pd.DataFrame(height_diffs, columns=['Absolute', 'Percentage'])
df.to_csv(os.path.join(result_path, 'height_diffs.txt'), index=False, float_format='%.2f')
