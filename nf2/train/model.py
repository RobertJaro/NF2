import torch
from torch import nn
from torch.nn import Embedding


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class HeightMappingModel(nn.Module):

    def __init__(self, in_coords, dim, positional_encoding=True):
        super().__init__()
        if positional_encoding:
            posenc = PositionalEncoding(8, 20)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(4)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 1)
        self.activation = Sine()

    def forward(self, x, height_range):
        input_coords = x
        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        z_coords = torch.sigmoid(self.d_out(x)) * (height_range[:, 1:2] - height_range[:, 0:1]) + height_range[:, 0:1]
        # shifted_z_coords = input_coords[:, 2:3] * (1 + z_shift) # max shift in dependence of height estimate
        output_coords = torch.cat([input_coords[:, :2], z_coords], -1)
        return output_coords

class BModel(nn.Module):

    def __init__(self, in_coords, out_values, dim, pos_encoding=False):
        super().__init__()
        if pos_encoding:
            posenc = PositionalEncoding(8, 20)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_values)
        self.activation = Sine()  # torch.tanh

    def forward(self, x):
        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        return x

class VectorPotentialModel(nn.Module):

    def __init__(self, in_coords, dim, pos_encoding=False):
        super().__init__()
        if pos_encoding:
            posenc = PositionalEncoding(8, 20)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 3)
        self.activation = Sine()  # torch.tanh

    def forward(self, x):
        coord = x
        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        a = self.d_out(x)
        #
        jac_matrix = jacobian(a, coord)
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
        #
        return b

class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, num_samples, in_features)
        Outputs:
            out: (batch, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(dim=-1)  # (num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (num_rays, num_samples, 2*num_freqs*in_features)
        return out


# TODO
class CartesianToAzimuthalTransform(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, vector, observer_coordinate):
        return get_field_inclination_azimuth(vector, observer_coordinate)


def get_field_inclination_azimuth(vector, observer_coordinate):
    projected = cartesian_to_image(vector, observer_coordinate)
    fld = torch.norm(projected, dim=-1)
    inc = torch.acos(projected[:, 2] / fld)
    azm = torch.atan2(projected[:, 1], projected[:, 0])
    return torch.stack([fld, inc, azm], -1)


def cartesian_to_image(vector, observer_coordinate):
    lat, lon = observer_coordinate[:, 0], observer_coordinate[:, 1]

    a11 = torch.cos(lon)
    a12 = 0
    a13 = - torch.sin(lon)
    a21 = -torch.sin(lat) * torch.sin(lon)
    a22 = torch.cos(lat)
    a23 = - torch.sin(lat) * torch.cos(lon)
    a31 = torch.cos(lat) * torch.sin(lon)
    a32 = torch.sin(lat)
    a33 = torch.cos(lat) * torch.cos(lon)

    a_matrix = torch.stack([a11, a12, a13, a21, a22, a23, a31, a32, a33], dim=-1).reshape(-1, 3, 3)
    ai_matrix = torch.inverse(a_matrix)

    image_vector = torch.matmul(ai_matrix, vector.unsqueeze(-1)).squeeze(-1)

    return image_vector

def calculate_current(b, coords):
    jac_matrix = jacobian(b, coords)
    dBx_dx = jac_matrix[:, 0, 0]
    dBy_dx = jac_matrix[:, 1, 0]
    dBz_dx = jac_matrix[:, 2, 0]
    dBx_dy = jac_matrix[:, 0, 1]
    dBy_dy = jac_matrix[:, 1, 1]
    dBz_dy = jac_matrix[:, 2, 1]
    dBx_dz = jac_matrix[:, 0, 2]
    dBy_dz = jac_matrix[:, 1, 2]
    dBz_dz = jac_matrix[:, 2, 2]
    #
    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    #
    j = torch.stack([rot_x, rot_y, rot_z], -1)
    return j

def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix