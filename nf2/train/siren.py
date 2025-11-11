import numpy as np
import torch
from torch import nn
from torch.nn import Identity
from torch.nn.functional import linear

from nf2.train.encoding import PositionalEncoding


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
    def __init__(self, in_dim, out_dim, dim=256, n_layers=8, w0=1., encoding_config=None, skip_layers=(2, 5), **kwargs):
        super().__init__()

        encoding_config = {'type': 'default', 'w0': 10.} if encoding_config is None else encoding_config
        encoding_type = encoding_config.pop('type', 'default')

        if encoding_type == "default":
            self.posenc = SirenLayer(in_dim=in_dim, out_dim=dim, is_first=True, **encoding_config)
            posenc_dim = dim
        elif encoding_type == "positional":
            self.posenc = PositionalEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.d_output
        elif encoding_type == "identity":
            self.posenc = Identity()
            posenc_dim = in_dim
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")

        self.num_layers = n_layers
        self.dim_hidden = dim
        self.skip_layers = skip_layers

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=posenc_dim, out_dim=dim, w0=w0)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            if i in self.skip_layers:
                # this layer will receive [h, skip_ref], width increases by posenc_dim
                in_d = dim + posenc_dim
            else:
                in_d = dim
            layer = SirenLayer(in_dim=in_d, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # initialize the output layer
        self.out_layer = SirenLayer(in_dim=dim, out_dim=out_dim, w0=w0, activation=nn.Identity())

    def forward(self, inp):
        inp_encoded = self.posenc(inp)  # apply positional encoding
        x = self.in_layer(inp_encoded)

        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                x = torch.cat([x, inp_encoded], dim=-1)
                x = layer(x)               # layer expects dim + ref_dim
            else:
                x = layer(x)               # standard SIREN layer

        x = self.out_layer(x)
        return x

    def step(self, global_step):
        if hasattr(self.posenc, 'step'):
            self.posenc.step(global_step)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
