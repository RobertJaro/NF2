import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1.0, c=6.0, is_first=False, use_bias=True):
        super().__init__()
        self.dim_in = in_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0)

    def init_(self, weight, bias, c, w0):
        w_std = (1 / self.dim_in) if self.is_first else (np.sqrt(c / self.dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        return self.activation(linear(x, self.weight, self.bias))


class SirenModel(nn.Module):
    """SIREN network used for all trainable NF2 field models."""

    def __init__(self, in_dim=3, out_dim=3, dim=512, n_layers=8, w0=1.0, w0_init=5.0, **kwargs):
        super().__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim
        self.in_layer = SirenLayer(in_dim=in_dim, out_dim=dim, w0=w0_init, is_first=True)
        self.layers = nn.ModuleList([
            SirenLayer(in_dim=dim, out_dim=dim, w0=w0)
            for _ in range(n_layers - 1)
        ])
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, coords):
        x = self.in_layer(coords)
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)


class BModel(SirenModel):
    """Direct magnetic-field SIREN model."""

    def __init__(self, **kwargs):
        super().__init__(in_dim=3, out_dim=3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        b = super().forward(coords)
        out = {"b": b}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


class VectorPotentialModel(SirenModel):
    """Vector-potential SIREN model with B = curl(A)."""

    def __init__(self, **kwargs):
        super().__init__(in_dim=3, out_dim=3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        a = super().forward(coords)
        jac_matrix = jacobian(a, coords)
        dAy_dx = jac_matrix[:, 1, 0]
        dAz_dx = jac_matrix[:, 2, 0]
        dAx_dy = jac_matrix[:, 0, 1]
        dAz_dy = jac_matrix[:, 2, 1]
        dAx_dz = jac_matrix[:, 0, 2]
        dAy_dz = jac_matrix[:, 1, 2]
        rot_x = dAz_dy - dAy_dz
        rot_y = dAx_dz - dAz_dx
        rot_z = dAy_dx - dAx_dy
        b = torch.stack([rot_x, rot_y, rot_z], -1)

        out = {"b": b, "a": a}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


def calculate_current(b, coords, jac_matrix=None):
    jac_matrix = jacobian(b, coords) if jac_matrix is None else jac_matrix
    return calculate_current_from_jacobian(jac_matrix)


def calculate_current_from_jacobian(jac_matrix, f=torch):
    dBx_dx = jac_matrix[..., 0, 0]
    dBy_dx = jac_matrix[..., 1, 0]
    dBz_dx = jac_matrix[..., 2, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBy_dy = jac_matrix[..., 1, 1]
    dBz_dy = jac_matrix[..., 2, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dz = jac_matrix[..., 2, 2]

    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    return f.stack([rot_x, rot_y, rot_z], -1)


def jacobian(output, coords):
    jac_matrix = [
        torch.autograd.grad(
            output[:, i],
            coords,
            grad_outputs=torch.ones_like(output[:, i]).to(output),
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        for i in range(output.shape[1])
    ]
    return torch.stack(jac_matrix, dim=1)
