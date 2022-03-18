import os

import numpy as np
import torch
from astropy.io import fits
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage.io import imsave
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nf2.data.dataset import CubeDataset, ImageDataset


def load_cube(save_path, device=None, batch_size=1000, z=None, progress=False):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    block_shape = cube_shape if z is None else [cube_shape[0], cube_shape[1], z]
    ds = CubeDataset(cube_shape, state['spatial_normalization'], block_shape)
    coords = torch.tensor(ds.getCube([0, 0, 0])).view((-1, 3))

    cube = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(coords), batch_size=batch_size)
        iter = loader if not progress else tqdm(loader)
        for batch, in iter:
            batch = batch.to(device)
            cube += [model(batch).detach().cpu()]

        cube = torch.cat(cube).view((*block_shape, 3)).numpy()
    b = cube * state['normalization']
    return b


def load_shape(save_path, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    return state['cube_shape']


def load_slice(save_path, z=0, device=None, batch_size=1000):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state = torch.load(save_path, map_location=device)
    model = nn.DataParallel(state['model'])
    cube_shape = state['cube_shape']
    ds = ImageDataset(cube_shape, state['spatial_normalization'], z)

    cube = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size):
            batch.requires_grad = True
            batch = batch.to(device)
            cube += [model(batch).detach().cpu()]

        cube = torch.cat(cube).view((cube_shape[0], cube_shape[1], 3)).numpy()
    b = cube * state['normalization']
    return b


def save_fits(vec, path, prefix):
    hdu = fits.PrimaryHDU(vec[..., 0])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_Bx.fits' % prefix))
    hdu = fits.PrimaryHDU(vec[..., 1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_By.fits' % prefix))
    hdu = fits.PrimaryHDU(vec[..., 2])
    hdul = fits.HDUList([hdu])
    hdul.writeto(os.path.join(path, '%s_Bz.fits' % prefix))


def save_slice(b, file_path, v_min_max=None):
    v_min_max = np.abs(b).max() if v_min_max is None else v_min_max
    b = Normalize(vmin=-v_min_max, vmax=v_min_max)(b)
    color_mapped = cm.get_cmap('gray')(np.flip(b.transpose(), 0))[..., :3]
    imsave(file_path, color_mapped)
