import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear


class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = in_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenModel(nn.Module):
    def __init__(self, in_dim, out_dim, dim=512, n_layers=8,
                 w0=1., w0_initial=30., **kwargs):
        super().__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=in_dim, out_dim=dim, w0=w0_initial, is_first=True)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            layer = SirenLayer(in_dim=dim, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        # initialize the output layer
        self.out_layer = SirenLayer(in_dim=dim, out_dim=out_dim, w0=w0, activation=nn.Identity())

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        x = self.out_layer(x)
        return x


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
