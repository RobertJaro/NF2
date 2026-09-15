import numpy as np
import torch
from astropy.nddata import block_replicate
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nf2.loader.base import DEFAULT_NUM_WORKERS


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


# calculate potential field using greens function method
def get_potential_field(b_n, height, batch_size=2048, strides=(1, 1, 1), progress=True):
    if isinstance(strides, int): strides = (strides, strides, strides)
    cube_shape = (*b_n.shape, height)
    batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size
    coords = [np.stack(
        np.mgrid[
            :cube_shape[0]:strides[0],
            :cube_shape[1]:strides[1],
            :cube_shape[2]:strides[2],
        ],
        -1,
    )]
    potential, = compute_scalar_potential(
        coords,
        cube_shape,
        b_n,
        batch_size=batch_size,
        progress=progress,
    )
    if strides != (1, 1, 1):
        potential = block_replicate(potential, strides, conserve_sum=False)
    b = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)
    return b


def _make_boundary_face_coords(
        cube_shape, only_top=False, include_derivative_stencil=False):
    nx, ny, nz = cube_shape
    face_specs = [(2, nz - 1)]
    if not only_top:
        face_specs += [(0, 0), (0, nx - 1), (1, 0), (1, ny - 1)]

    face_coords = []
    for axis, boundary_index in face_specs:
        axis_coords = [np.arange(nx), np.arange(ny), np.arange(nz)]
        axis_coords[axis] = (
            np.arange(boundary_index - 1, boundary_index + 2)
            if include_derivative_stencil
            else np.array([boundary_index])
        )
        face_coords += [np.stack(np.meshgrid(*axis_coords, indexing='ij'), -1)]
    return face_coords, [axis for axis, _ in face_specs]


def _flatten_boundary_faces(face_coords, face_fields, axes):
    coords_flat = []
    fields_flat = []
    for coords, fields, axis in zip(face_coords, face_fields, axes):
        idx = [slice(None)] * 3
        idx[axis] = 1 if coords.shape[axis] == 3 else 0
        idx = tuple(idx)
        coords_flat += [coords[idx].reshape((-1, 3))]
        fields_flat += [fields[idx].reshape((-1, 3))]
    return np.concatenate(coords_flat), np.concatenate(fields_flat)


def get_potential_boundary(b_n, height, batch_size=None, only_top=False,
                           method='direct', progress=False, **kwargs):
    assert not np.any(np.isnan(b_n)), 'Invalid data value'
    if method.lower() not in {'direct', 'green', 'greens', 'fft'}:
        raise ValueError("Potential field method must be 'fft' or 'direct'.")

    if method == 'fft':
        field = get_fft_potential_field(b_n, int(height), **kwargs)
        coords, axes = _make_boundary_face_coords(field.shape[:3], only_top=only_top)
        fields = [field[tuple(np.moveaxis(coord, -1, 0))] for coord in coords]
        return _flatten_boundary_faces(coords, fields, axes)

    if batch_size is None:
        batch_size = int(1024 * 512 ** 2 / np.prod(b_n.shape))
    cube_shape = (*b_n.shape, height)
    coords, axes = _make_boundary_face_coords(
        cube_shape,
        only_top=only_top,
        include_derivative_stencil=True,
    )
    fields = compute_potential(
        coords,
        cube_shape,
        b_n,
        batch_size=batch_size,
        progress=progress,
        **kwargs,
    )
    return _flatten_boundary_faces(coords, fields, axes)

def compute_scalar_potential(coords, cube_shape, b_n, batch_size=2048, progress=False):
    flat_coords = np.concatenate([c.reshape(((-1, 3))) for c in coords])
    b_n = np.asarray(b_n)
    expected_shape = tuple(cube_shape[:2])
    if b_n.shape != expected_shape:
        raise ValueError(f'b_n must have shape {expected_shape}, got {b_n.shape}')
    b_n = b_n.reshape((-1)).astype(np.float32)

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
        loader = DataLoader(TensorDataset(flat_coords), batch_size=batch_size, num_workers=min(2, DEFAULT_NUM_WORKERS))
        loader = tqdm(loader, desc='Potential Field') if progress else loader
        for coord, in loader:
            coord = coord.to(device)
            p_batch = model(coord)
            potential += [p_batch.detach().cpu()]

    grid_sizes = [int(np.prod(c.shape[:-1])) for c in coords]
    potential_chunks = torch.cat(potential).split(grid_sizes)
    return [
        chunk.reshape(c.shape[:-1]).numpy()
        for chunk, c in zip(potential_chunks, coords)
    ]


def compute_potential(coords, cube_shape, b_n, batch_size=2048, progress=False):
    potentials = compute_scalar_potential(
        coords,
        cube_shape,
        b_n,
        batch_size=batch_size,
        progress=progress,
    )
    fields = [
        -1 * np.stack(np.gradient(p, edge_order=2), axis=-1)
        for p in potentials
    ]
    return fields

def get_fft_potential_field(Bz0, Nz, scale=1, alpha=0):
    Nx, Ny = Bz0.shape
    z = np.arange(Nz) * scale

    fftBz0 = np.fft.fft2(Bz0)

    # Use FFT-native frequency bins so odd/even sizes are handled correctly.
    kx = (2 * np.pi) * np.fft.fftfreq(Nx)[:, None]
    ky = (2 * np.pi) * np.fft.fftfreq(Ny)[None, :]

    k2 = kx * kx + ky * ky
    w2 = k2 - alpha ** 2
    wabs = np.sqrt(w2 + 0j)

    HxB = np.zeros((Nx, Ny), dtype=complex)
    HyB = np.zeros((Nx, Ny), dtype=complex)
    HzB = np.ones((Nx, Ny), dtype=complex)

    HxB[0, 0] = -1j
    HyB[0, 0] = -1j

    mask = k2 != 0
    HxB[mask] = (-1j * kx * wabs + 1j * alpha * ky)[mask] / k2[mask]
    HyB[mask] = (-1j * ky * wabs - 1j * alpha * kx)[mask] / k2[mask]

    b = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32, order="F")

    for i in range(Nz):
        fftBz = fftBz0 * np.exp(-wabs * z[i])

        b[:, :, i, 0] = np.fft.ifft2(fftBz * HxB).real
        b[:, :, i, 1] = np.fft.ifft2(fftBz * HyB).real
        b[:, :, i, 2] = np.fft.ifft2(fftBz * HzB).real
    return b
