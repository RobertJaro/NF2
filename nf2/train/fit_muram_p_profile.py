import argparse
import os

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from nf2.loader.muram import MURaMSnapshot

from nf2.train.model import GenericModel

parser = argparse.ArgumentParser()
parser.add_argument('--muram_source_path', type=str, required=True)
parser.add_argument('--iteration', type=int, required=True)
parser.add_argument('--base_height', type=int, default=116)
parser.add_argument('--out_path', type=str, required=True)
args = parser.parse_args()

out_path = args.out_path
os.makedirs(out_path, exist_ok=True)

snapshot = MURaMSnapshot(args.muram_source_path, args.iteration)

Mm_per_pixel = snapshot.ds[2].to_value(u.Mm / u.pix)
base_height = args.base_height
# max_height = int(100 / Mm_per_pixel) + base_height
# min_height =  base_height - int(5 / Mm_per_pixel)
muram_p = snapshot.P #u.G ** 2 * u.cm ** 2
muram_rho = snapshot.rho #u.g / u.cm ** 3

gauss_per_dB = 2500
Mm_per_ds = .36 * 320
# cm_per_ds = 1e2 * meters_per_ds

muram_p = muram_p.mean((0, 1)) / (gauss_per_dB ** 2)
muram_log_p = np.log10(muram_p)

coords = np.mgrid[:muram_log_p.shape[0]] - base_height # set the base height to 0
coords =  coords * (Mm_per_pixel / Mm_per_ds)

p_model = GenericModel(1, 1, 8, 2)
optimizer = torch.optim.Adam(p_model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_p_tensor = torch.tensor(muram_log_p[:, None], dtype=torch.float32).to(device)
coords_tensor = torch.tensor(coords[:, None], dtype=torch.float32).to(device)
p_model.to(device)

p_model.train()

for epoch in range(int(1e5)):
    optimizer.zero_grad()
    #
    pred_log_p = p_model.forward(coords_tensor)
    loss = (pred_log_p - log_p_tensor).pow(2).mean()
    loss.backward()
    optimizer.step()
    #
    if (epoch + 1) % 1e4 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
        fig, ax = plt.subplots()
        #
        ax.plot(log_p_tensor.cpu().detach().numpy(), coords_tensor.cpu().detach().numpy(), label='MURaM')
        ax.plot(pred_log_p.cpu().detach().numpy(), coords_tensor.cpu().detach().numpy(), label='Predicted')
        #
        ax.legend()
        plt.savefig(os.path.join(out_path, f'p_profile_{epoch + 1:06d}.jpg'))
        plt.close(fig)


torch.save(p_model, os.path.join(out_path, f'log_p_model.pt'))
