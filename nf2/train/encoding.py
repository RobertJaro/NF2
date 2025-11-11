from typing import Iterable

import torch
from torch import nn


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
            encoded = x[:, None, i:i+1] * torch.pi * frequencies
            encoded = encoded.reshape(x.shape[0], -1)
            encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded