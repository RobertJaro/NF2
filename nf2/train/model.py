import numpy as np
import torch
from astropy import units as u
from torch import nn
from torch.distributions import Normal

from nf2.data.util import cartesian_to_spherical, spherical_to_cartesian


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class RadialTransformModel(nn.Module):

    def __init__(self, in_coords, dim, positional_encoding=True, ds_ids=[]):
        super().__init__()
        if positional_encoding:
            posenc = GaussianPositionalEncoding(d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(4)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 1)
        self.activation = Sine()
        self.ds_ids = ds_ids
        self.observer_transformer = ObserverTransformer()

    def forward(self, coords, batch):
        for ds_id in self.ds_ids:
            transformed_coords = self.transform(batch[ds_id]['coords'], batch[ds_id]['obs_coords'],
                                                batch[ds_id]['height_range'])
            batch[ds_id]['coords'] = transformed_coords
            batch[ds_id]['original_coords'] = coords

    def transform(self, coords, obs_coords, height_range, **kwargs):
        coords = self.observer_transformer.transform(coords, obs_coords)

        x = self.activation(self.d_in(coords))
        for l in self.linear_layers:
            x = self.activation(l(x))
        z_coords = torch.sigmoid(self.d_out(x)) * (height_range[:, 1:2] - height_range[:, 0:1]) + height_range[:, 0:1]
        output_coords = torch.cat([coords[:, :2], z_coords], -1)

        output_coords = self.observer_transformer.inverse_transform(output_coords, obs_coords)

        return output_coords


class GenericModel(nn.Module):

    def __init__(self, in_coords, out_coords, coord_range=None, ds_per_pixel=None, dim=256, n_layers=8, encoding_config=None, activation='sine'):
        super().__init__()
        encoding_type = encoding_config['type'] if encoding_config is not None else 'none'
        if encoding_type == 'none':
            self.d_in = nn.Linear(in_coords, dim)
        elif encoding_type == 'positional':
            num_freqs = encoding_config['num_freqs'] if 'num_freqs' in encoding_config else 128
            min_freq = encoding_config['min_freq'] if 'min_freq' in encoding_config else -2
            max_freq = encoding_config['max_freq'] if 'max_freq' in encoding_config else 8
            posenc = PositionalEncoding(in_coords, num_freqs=num_freqs, min_freq=min_freq, max_freq=max_freq)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding_type == 'gaussian':
            posenc = GaussianPositionalEncoding(in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding_type == 'periodic':
            posenc = PeriodicEncoding(in_coords, coord_range=coord_range, ds_per_pixel=ds_per_pixel)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding_type == 'periodic_gaussian':
            scale = encoding_config.pop('scale', 4.0)
            periodic_enc = PeriodicEncoding(in_coords, coord_range=coord_range, ds_per_pixel=ds_per_pixel)
            posenc = GaussianPositionalEncoding(periodic_enc.d_output, scale=scale)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(periodic_enc, posenc, d_in)
        else:
            raise NotImplementedError(f'Unknown encoding {encoding_type}')
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_coords)
        activation_mapping = {'relu': nn.ReLU, 'swish': Swish, 'tanh': nn.Tanh, 'sine': Sine}
        activation_f = activation_mapping[activation]
        self.in_activation = activation_f()
        self.activations = nn.ModuleList([activation_f() for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        x = self.d_out(x)
        return x


class ScalingModel(GenericModel):

    def __init__(self):
        super().__init__(in_coords=1, out_coords=1, dim=8, n_layers=2)

    def forward(self, z):
        return 10 ** super().forward(z)

class GenericDomainModel(nn.Module):

    def __init__(self, Mm_per_ds, in_dim=3, out_dim=3, window_type='sigmoid', spherical=False, range_config=None,
                 overlap_width=1.0,
                 **model_config):
        # use default range config if not provided
        self.spherical = spherical
        # use default range config if not provided
        range_config = self._load_default_range_config(spherical) if range_config is None else range_config
        domain_range = self._load_range(range_config, Mm_per_ds, spherical)
        # normalize overlap width
        overlap = overlap_width / Mm_per_ds
        #
        super().__init__()
        self.models = nn.ModuleList([GenericModel(in_dim, out_dim, **model_config) for _ in range_config])
        self.domain_range = nn.Parameter(torch.tensor(domain_range, dtype=torch.float32), requires_grad=False)
        self.overlap = nn.Parameter(torch.tensor(overlap, dtype=torch.float32), requires_grad=False)
        assert window_type in ['sigmoid', 'step'], f'Unknown window type {window_type}. Choose from [sigmoid, step]'
        self.window_type = window_type
        self.streams = [torch.cuda.Stream() for _ in range(len(self.models))]

    def _load_range(self, range_config, Mm_per_ds, spherical):
        # normalize range config
        domain_range = [(rc['start'], rc['end']) for rc in range_config]
        domain_range = [[np.nan if r is None else r for r in rc] for rc in domain_range]  # replace None with np.nan
        if spherical:  # add solar radius offset
            domain_range = np.array(domain_range) + (1 * u.solRad).to_value(u.Mm)
        domain_range = np.array(domain_range) / Mm_per_ds
        print('RANGE CONFIG', domain_range)
        return domain_range

    def _load_default_range_config(self, spherical):
        if spherical:
            return [{'start': None, 'end': 10},
                    {'start': 10, 'end': 100},
                    {'start': 100, 'end': None}, ]
        else:
            return [{'start': None, 'end': 10},
                    {'start': 10, 'end': 50},
                    {'start': 50, 'end': None},
                    ]

    def forward(self, coords):
        z = coords[:, 2:3] if not self.spherical else torch.norm(coords[..., :3], dim=-1)[..., None]

        outputs = []
        for i, model in enumerate(self.models):
            start, end = self.domain_range[i]
            overlap = self.overlap
            if self.window_type == 'sigmoid':
                left = torch.sigmoid((z - start) * 2 / overlap) if not torch.isnan(start) else 1
                right = torch.sigmoid((z - end) * 2 / overlap) if not torch.isnan(end) else 0
                window = left - right
            elif self.window_type == 'step':
                left_center = (z >= start) if not torch.isnan(start) else torch.ones_like(z, dtype=torch.bool)
                right_center = (z < end) if not torch.isnan(end) else torch.ones_like(z, dtype=torch.bool)
                center = left_center & right_center
                left_overlap = (z >= start - overlap) & (z < start) if not torch.isnan(start) else torch.zeros_like(z,
                                                                                                                    dtype=torch.bool)
                right_overlap = (z >= end) & (z < end + overlap) if not torch.isnan(end) else torch.zeros_like(z,
                                                                                                               dtype=torch.bool)
                overlap = left_overlap | right_overlap
                window = center.float() + 0.5 * overlap.float()
            else:
                raise NotImplementedError(f'Unknown window type {self.window_type}')
            #
            with torch.cuda.stream(self.streams[i]):
                out = model(coords)
            # combine outputs with window function
            outputs.append(out * window)

        torch.cuda.synchronize()
        # sum outputs over domains
        out = torch.sum(torch.stack(outputs, -1), -1)
        return out


class MultiDomainModel(nn.Module):

    def __init__(self, Mm_per_ds, window_type='sigmoid', spherical=False, range_config=None, overlap_width=1.0,
                 **model_config):
        # use default range config if not provided
        self.spherical = spherical
        # use default range config if not provided
        range_config = self._load_default_range_config(spherical) if range_config is None else range_config
        domain_range = self._load_range(range_config, Mm_per_ds, spherical)
        # normalize overlap width
        overlap = overlap_width / Mm_per_ds
        #
        super().__init__()
        self.models = nn.ModuleList([self._init_model(rc['model_type'], **model_config) for rc in range_config])
        self.domain_range = nn.Parameter(torch.tensor(domain_range, dtype=torch.float32), requires_grad=False)
        self.overlap = nn.Parameter(torch.tensor(overlap, dtype=torch.float32), requires_grad=False)
        assert window_type in ['sigmoid', 'step'], f'Unknown window type {window_type}. Choose from [sigmoid, step]'
        self.window_type = window_type

    def _load_range(self, range_config, Mm_per_ds, spherical):
        # normalize range config
        domain_range = [(rc['start'], rc['end']) for rc in range_config]
        domain_range = [[np.nan if r is None else r for r in rc] for rc in domain_range]  # replace None with np.nan
        if spherical:  # add solar radius offset
            domain_range = np.array(domain_range) + (1 * u.solRad).to_value(u.Mm)
        domain_range = np.array(domain_range) / Mm_per_ds
        print('RANGE CONFIG', domain_range)
        return domain_range

    def _load_default_range_config(self, spherical):
        if spherical:
            return [{'model_type': 'vector_potential', 'start': None, 'end': 10},
                    {'model_type': 'vector_potential', 'start': 10, 'end': 100},
                    {'model_type': 'potential', 'start': 100, 'end': None}, ]
        else:
            return [{'model_type': 'vector_potential', 'start': None, 'end': 10},
                    {'model_type': 'vector_potential', 'start': 10, 'end': 50},
                    {'model_type': 'vector_potential', 'start': 50, 'end': None},
                    ]

    def _init_model(self, model_type, **model_config):
        # create models and parameters
        if model_type == 'vector_potential':
            model_class = VectorPotentialModel
        elif model_type == 'b':
            model_class = BModel
        elif model_type == 'b_scaled':
            model_class = BScaledModel
        elif model_type == 'potential':
            model_class = PotentialModel
        elif model_type == 'radial':
            model_class = RadialModel
        else:
            raise NotImplementedError(f'Unknown model {model_type}')
        return model_class(**model_config)

    def forward(self, coords, compute_jacobian=True):
        z = coords[:, 2:3] if not self.spherical else torch.norm(coords[..., :3], dim=-1)[..., None]

        outputs = []
        combined_out = {}
        for i, model in enumerate(self.models):
            start, end = self.domain_range[i]
            overlap = self.overlap
            if self.window_type == 'sigmoid':
                left = torch.sigmoid((z - start) * 2 / overlap) if not torch.isnan(start) else 1
                right = torch.sigmoid((z - end) * 2 / overlap) if not torch.isnan(end) else 0
                window = left - right
            elif self.window_type == 'step':
                left_center = (z >= start) if not torch.isnan(start) else torch.ones_like(z, dtype=torch.bool)
                right_center = (z < end) if not torch.isnan(end) else torch.ones_like(z, dtype=torch.bool)
                center = left_center & right_center
                left_overlap = (z >= start - overlap) & (z < start) if not torch.isnan(start) else torch.zeros_like(z,
                                                                                                                    dtype=torch.bool)
                right_overlap = (z >= end) & (z < end + overlap) if not torch.isnan(end) else torch.zeros_like(z,
                                                                                                               dtype=torch.bool)
                overlap = left_overlap | right_overlap
                window = center.float() + 0.5 * overlap.float()
            else:
                raise NotImplementedError(f'Unknown window type {self.window_type}')
            #
            out = model(coords)
            if 'b_height_scaling' in out:
                combined_out['b_height_scaling'] = out['b_height_scaling']
            # combine outputs with window function
            outputs.append(out['b'] * window)

        # sum outputs over domains
        b = torch.sum(torch.stack(outputs, -1), -1)

        combined_out['b'] = b
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            combined_out['jac_matrix'] = jac_matrix

        return combined_out


class BDomainModel(GenericDomainModel):

    def forward(self, coords, compute_jacobian=True):
        b = super().forward(coords)

        out = {'b': b}
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out['jac_matrix'] = jac_matrix
        return out


class VectorPotentialDomainModel(GenericDomainModel):

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
        out = {'b': b, 'a': a}

        # compute jacobian
        if compute_jacobian:
            self.zero_grad()  # does this do anything?
            jac_matrix = jacobian(out['b'], coords)
            out['jac_matrix'] = jac_matrix
        return out


class BModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        b = super().forward(coords)
        out_dict = {'b': b}
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        return out_dict


class BScaledModel(nn.Module):

    def __init__(self, spherical=False, **model_kwargs):
        super().__init__()
        self.b_model = GenericModel(3, 3, **model_kwargs)
        self.b_height_scaling_model = ScalingModel()
        self.spherical = spherical

    def forward(self, coords, compute_jacobian=True):
        b = self.b_model(coords)

        z = coords[:, 2:3] if not self.spherical else torch.norm(coords[..., :3], dim=-1, keepdim=True)
        b_height_scaling = self.b_height_scaling_model(z)

        out_dict = {'b': b, 'b_height_scaling': b_height_scaling}
        #
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        #
        return out_dict


class VectorPotentialScaledModel(nn.Module):

    def __init__(self, spherical=False, **model_kwargs):
        super().__init__()
        self.a_model = GenericModel(3, 4, **model_kwargs)

    def forward(self, coords, compute_jacobian=True):
        x = self.a_model(coords)
        a = x[:, :3] * 10 ** x[:, 3:4]
        #
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
        out_dict = {'b': b, 'a': a}
        #
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        #
        return out_dict


class VectorPotentialModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        a = super().forward(coords)
        #
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
        out_dict = {'b': b, 'a': a}
        #
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        #
        return out_dict


class PotentialModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 1, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        phi = super().forward(coords)
        #
        jac_matrix = jacobian(phi, coords)
        dphi_dx = jac_matrix[:, 0, 0]
        dphi_dy = jac_matrix[:, 0, 1]
        dphi_dz = jac_matrix[:, 0, 2]
        b = -torch.stack([dphi_dx, dphi_dy, dphi_dz], -1)
        out_dict = {'b': b, 'phi': phi}
        #
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        #
        return out_dict


class RadialModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 1, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        b_scale = super().forward(coords)
        r_unit = coords / torch.norm(coords, dim=-1, keepdim=True)

        b = b_scale * r_unit
        out_dict = {'b': b}

        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix

        return out_dict


class PressureScaledModel(nn.Module):

    def __init__(self, Mm_per_ds, domain_range=5, overlap=2, cutoff=False, **kwargs):
        super().__init__()
        self.domain_range = domain_range / Mm_per_ds
        self.overlap = overlap / Mm_per_ds
        self.p_height_scaling_model = ScalingModel()
        self.p_model = GenericModel(3, 1, **kwargs)
        self.cutoff = cutoff


    def forward(self, coords, compute_jacobian=True):
        p = 10 ** self.p_model(coords)
        z = coords[:, 2:3]

        if self.cutoff:
            # apply height mask
            end = self.domain_range
            overlap = self.overlap
            #
            right = torch.sigmoid((z - end) * 2 / overlap)
            window = 1 - right
            p = p * window
        #
        p_height_scaling = self.p_height_scaling_model(z)
        #
        out_dict = {'p': p, 'p_height_scaling': p_height_scaling}
        if compute_jacobian:
            jac_matrix = jacobian(p, coords)
            out_dict['jac_matrix'] = jac_matrix
            # compute gradient P
            dP_dx = jac_matrix[:, 0, 0]
            dP_dy = jac_matrix[:, 0, 1]
            dP_dz = jac_matrix[:, 0, 2]
            grad_P = torch.stack([dP_dx, dP_dy, dP_dz], -1)
            out_dict['grad_P'] = grad_P
        return out_dict

class MagnetoStaticModel(nn.Module):

    def __init__(self, Mm_per_ds, cutoff=False, **kwargs):
        super().__init__()
        self.b_model = BScaledModel(**kwargs)
        self.p_model = PressureScaledModel(Mm_per_ds, cutoff=cutoff, **kwargs)


    def forward(self, coords, compute_jacobian=True):
        b_dict = self.b_model(coords)
        p_dict = self.p_model(coords)
        out_dict = {**b_dict, **p_dict}
        if compute_jacobian:
            b = b_dict['b']
            jac_b_matrix = jacobian(b, coords)
            p = p_dict['p']
            jac_p_matrix = jacobian(p, coords)
            jac_matrix = torch.cat([jac_b_matrix, jac_p_matrix], -2)
            out_dict['jac_matrix'] = jac_matrix

        return out_dict


class PositionalEncoding(nn.Module):

    def __init__(self, in_features, num_freqs=128, min_freq=-2, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(min_freq, max_freq, num_freqs) * torch.pi
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.d_output = in_features * (1 + num_freqs * 2)

    def forward(self, x):
        encoded = torch.einsum('...i,j->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded), x], -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=128, scale=2.0 ** 2):
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
        coord_range[..., 1] = coord_range[..., 1] + ds_per_pixel # add one pixel --> [0, 2pi] = [0, n_pix + 1]
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

class ObserverTransformer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, coords, obs_coord):
        return self.transform(coords, obs_coord)

    def transform(self, coords, obs_coord):
        # transform coords to observer frame
        coords = coords - obs_coord
        coords = cartesian_to_spherical(coords, f=torch)
        return coords

    def inverse_transform(self, coords, obs_coord):
        # transform coords to solar frame
        coords = spherical_to_cartesian(coords, f=torch)
        coords = coords + obs_coord
        return coords


def image_to_spherical_matrix(lon, lat, latc, lonc, pAng, sin=np.sin, cos=np.cos):
    a11 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
    a12 = sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
    a13 = -cos(latc) * sin(lon - lonc)
    a21 = -sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - cos(lat) * cos(
        latc) * sin(pAng)
    a22 = sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + cos(lat) * cos(
        latc) * cos(pAng)
    a23 = -cos(latc) * sin(lat) * cos(lon - lonc) + sin(latc) * cos(lat)
    a31 = cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(
        latc) * sin(pAng)
    a32 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(
        latc) * cos(pAng)
    a33 = cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)

    # a_matrix = np.stack([a11, a12, a13, a21, a22, a23, a31, a32, a33], axis=-1)
    a_matrix = np.stack([a31, a32, a33, a21, a22, a23, a11, a12, a13], axis=-1)
    a_matrix = a_matrix.reshape((*a_matrix.shape[:-1], 3, 3))
    return a_matrix


def calculate_current(b, coords, jac_matrix=None):
    jac_matrix = jacobian(b, coords) if jac_matrix is None else jac_matrix
    j = calculate_current_from_jacobian(jac_matrix)
    return j


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
    #
    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    #
    j = f.stack([rot_x, rot_y, rot_z], -1)
    return j


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix
