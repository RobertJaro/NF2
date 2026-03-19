from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from nf2.train.layers import SirenLayer


class PositionalEncoding(nn.Module):

    def __init__(self, in_dim, num_frequencies=32, min_frequencies=0, max_frequencies=8):
        super().__init__()
        num_frequencies = [num_frequencies] * in_dim if not isinstance(num_frequencies, Iterable) else num_frequencies
        min_frequencies = [min_frequencies] * in_dim if not isinstance(min_frequencies, Iterable) else min_frequencies
        max_frequencies = [max_frequencies] * in_dim if not isinstance(max_frequencies, Iterable) else max_frequencies

        assert len(num_frequencies) == in_dim, 'num_frequencies length must match input dimension (in_dim)'
        assert len(min_frequencies) == in_dim, 'd_min_frequencies length must match input dimension (in_dim)'
        assert len(max_frequencies) == in_dim, 'max_frequencies length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, min_freq, max_freq in zip(num_frequencies, min_frequencies, max_frequencies):
            f = 2 ** torch.linspace(min_freq, max_freq, num_freq)
            param = nn.Parameter(f[None, :, None], requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.d_output = sum([n * 2 for n in num_frequencies])

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = x[:, None, i:i + 1] * torch.pi * frequencies
            encoded = encoded.reshape(x.shape[0], -1)
            encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded


class MultispectralEncoding(nn.Module):

    def __init__(self, in_dim, num_dims=64, weights=30.0):
        super().__init__()
        num_dims = [num_dims] * in_dim if not isinstance(num_dims, Iterable) else num_dims
        weights = [weights] * in_dim if not isinstance(weights, Iterable) else weights

        assert len(num_dims) == in_dim, 'num_dims length must match input dimension (in_dim)'
        assert len(weights) == in_dim, 'weights length must match input dimension (in_dim)'

        layers = []
        for nd, w in zip(num_dims, weights):
            l = SirenLayer(in_dim=1, out_dim=nd, w0=w, is_first=True)
            layers.append(l)
        self.layers = nn.ModuleList(layers)

        self.d_output = sum(num_dims)

    def forward(self, x):
        encoded_coordinates = []
        for i, layer in enumerate(self.layers):
            coord = x[:, i:i + 1]
            encoded = layer(coord)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=64, scale=2.0 ** 1):
        super().__init__()
        dist = Normal(loc=0, scale=scale)
        frequencies = dist.sample([num_freqs, d_input])
        self.frequencies = nn.Parameter(2 * torch.pi * frequencies, requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = torch.einsum('...j,ij->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class PeriodicEncoding(nn.Module):

    def __init__(self, d_input, coord_range, ds_per_pixel):
        super().__init__()
        coord_range[..., 1] = coord_range[..., 1] + ds_per_pixel  # add one pixel --> [0, 2pi] = [0, n_pix + 1]
        self.coord_range = nn.Parameter(torch.tensor(coord_range, dtype=torch.float32), requires_grad=False)
        self.d_output = d_input + 2

    def forward(self, coord):
        scaled_x = (coord[..., 0:1] - self.coord_range[0, 0]) / (
                self.coord_range[0, 1] - self.coord_range[0, 0]) * 2 * torch.pi
        scaled_y = (coord[..., 1:2] - self.coord_range[1, 0]) / (
                self.coord_range[1, 1] - self.coord_range[1, 0]) * 2 * torch.pi
        encoded_coord = torch.cat([
            torch.sin(scaled_x), torch.cos(scaled_x),
            torch.sin(scaled_y), torch.cos(scaled_y),
            coord[..., 2:]], -1)
        return encoded_coord
