import numpy as np
import torch
from astropy.nddata import block_replicate
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PotentialModel(nn.Module):

    def __init__(self, b_n, r_p):
        super().__init__()
        self.register_buffer('b_n', b_n)
        self.register_buffer('r_p', r_p)
        c = np.zeros((1, 3))
        c[:, 2] = (1 / np.sqrt(2 * np.pi))
        c = torch.tensor(c, dtype=torch.float32, )
        self.register_buffer('c', c)

    def forward(self, coord):
        v1 = self.b_n[:, None]
        v2 = 2 * np.pi * ((-self.r_p[:, None] + coord[None, :] + self.c[None]) ** 2).sum(-1) ** 0.5
        potential = torch.sum(v1 / v2, dim=0)
        return potential


def get_potential(b_n, height, batch_size=2048, strides=(1, 1, 1), progress=True):
    cube_shape = (*b_n.shape, height)
    strides = (strides, strides, strides) if isinstance(strides, int) else strides
    b_n = b_n.reshape((-1,)).astype(np.float32)
    coords = np.stack(np.mgrid[:cube_shape[0]:strides[0], :cube_shape[1]:strides[1], :cube_shape[2]:strides[2]], -1).reshape((-1, 3))
    r_p = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], :1], -1).reshape((-1, 3))

    # torch code
    # r = (x * y, 3); coords = (x*y*z, 3), c = (1, 3)
    # --> (x * y, x * y * z, 3) --> (x * y, x * y * z) --> (x * y * z)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        b_n = torch.tensor(b_n, dtype=torch.float32, )
        r_p = torch.tensor(r_p, dtype=torch.float32, )
        model = nn.DataParallel(PotentialModel(b_n, r_p)).to(device)

        coords = torch.tensor(coords, dtype=torch.float32)
        potential = []
        loader = DataLoader(TensorDataset(coords), batch_size=batch_size, num_workers=8)
        it = tqdm(loader, desc='Potential Field') if progress else loader
        for coord, in it:
            coord = coord.to(device)
            p_batch = model(coord)
            potential += [p_batch]

    potential = torch.cat(potential).view(cube_shape).cpu().numpy()
    if strides != (1, 1, 1):
        potential = block_replicate(potential, strides, conserve_sum=False)
    return potential


def get_potential_boundary(b_n, height, batch_size=2048):
    assert not np.any(np.isnan(b_n)), 'Invalid data value'

    cube_shape = (*b_n.shape, height)

    b_n = b_n.reshape((-1)).astype(np.float32)
    coords = [np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], cube_shape[2] - 2:cube_shape[2] + 1], -1),
              np.stack(np.mgrid[:cube_shape[0], -1:2, :cube_shape[2]], -1),
              np.stack(np.mgrid[:cube_shape[0], cube_shape[1] - 2:cube_shape[1] + 1, :cube_shape[2]], -1),
              np.stack(np.mgrid[-1:2, :cube_shape[1], :cube_shape[2]], -1),
              np.stack(np.mgrid[cube_shape[0] - 2:cube_shape[0] + 1, :cube_shape[1], :cube_shape[2]], -1), ]
    coords_shape = [c.shape[:-1] for c in coords]
    flat_coords = np.concatenate([c.reshape(((-1, 3))) for c in coords])

    r_p = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], :1], -1).reshape((-1, 3))

    # torch code
    # r = (x * y, 3); coords = (x*y*z, 3), c = (1, 3)
    # --> (x * y, x * y * z, 3) --> (x * y, x * y * z) --> (x * y * z)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        b_n = torch.tensor(b_n, dtype=torch.float32, )
        r_p = torch.tensor(r_p, dtype=torch.float32, )
        model = nn.DataParallel(PotentialModel(b_n, r_p, )).to(device)

        flat_coords = torch.tensor(flat_coords, dtype=torch.float32, )

        potential = []
        for coord, in tqdm(DataLoader(TensorDataset(flat_coords), batch_size=batch_size, num_workers=2),
                           desc='Potential Boundary'):
            coord = coord.to(device)
            p_batch = model(coord)
            potential += [p_batch.cpu()]

    potential = torch.cat(potential).numpy()
    idx = 0
    fields = []
    for s in coords_shape:
        p = potential[idx:idx + np.prod(s)].reshape(s)
        b = - 1 * np.stack(np.gradient(p, axis=[0, 1, 2], edge_order=2), axis=-1)
        fields += [b]
        idx += np.prod(s)

    fields = [fields[0][:, :, 1].reshape((-1, 3)),
              fields[1][:, 1, :].reshape((-1, 3)), fields[2][:, 1, :].reshape((-1, 3)),
              fields[3][1, :, :].reshape((-1, 3)), fields[4][1, :, :].reshape((-1, 3))]
    coords = [coords[0][:, :, 1].reshape((-1, 3)),
              coords[1][:, 1, :].reshape((-1, 3)), coords[2][:, 1, :].reshape((-1, 3)),
              coords[3][1, :, :].reshape((-1, 3)), coords[4][1, :, :].reshape((-1, 3))]
    return np.concatenate(coords), np.concatenate(fields)
