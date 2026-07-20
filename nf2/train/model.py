import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear

SOLAR_RADIUS_Mm = 695.7


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

    def __init__(self, in_dim=3, out_dim=3, dim=256, n_layers=8, w0=1.0, w0_init=5.0, **kwargs):
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
        b = curl(a, coords)

        out = {"b": b, "a": a}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


class ScaledPotentialModel(SirenModel):
    """Scalar-potential model with radial coordinate and potential power-law envelopes."""

    def __init__(self, radial_power=1.0, coordinate_radial_power=4.0,
                 base_radius=None, Mm_per_ds=None, eps=1e-6, **kwargs):
        super().__init__(in_dim=3, out_dim=1, **kwargs)
        self.requires_grad_forward = True
        if base_radius is None:
            if Mm_per_ds is None:
                raise ValueError("ScaledPotentialModel requires 'Mm_per_ds' when 'base_radius' is not set.")
            base_radius = SOLAR_RADIUS_Mm / Mm_per_ds
        if base_radius <= 0:
            raise ValueError("base_radius must be positive.")
        self.radial_power = radial_power
        self.coordinate_radial_power = coordinate_radial_power
        self.base_radius = base_radius
        self.eps = eps

    def forward(self, coords, compute_jacobian=True):
        b, phi, network_coords, coordinate_scale = self._scaled_potential_field(coords)

        out = {"b": b, "phi": phi, "network_coords": network_coords, "coordinate_scale": coordinate_scale}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out

    def _scaled_potential_field(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.eps)
        normalized_radius = radius / self.base_radius
        coordinate_scale = normalized_radius.pow(-self.coordinate_radial_power)
        network_coords = coords * coordinate_scale
        phi = SirenModel.forward(self, network_coords)
        phi = phi * normalized_radius.pow(-self.radial_power)
        b = -gradient(phi, coords)
        return b, phi, network_coords, coordinate_scale


class ScaledVectorPotentialModel(VectorPotentialModel):
    """Vector-potential model with radial coordinate and A power-law envelopes."""

    def __init__(self, radial_power=2.0, coordinate_radial_power=4.0,
                 base_radius=None, Mm_per_ds=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        if base_radius is None:
            if Mm_per_ds is None:
                raise ValueError("ScaledVectorPotentialModel requires 'Mm_per_ds' when 'base_radius' is not set.")
            base_radius = SOLAR_RADIUS_Mm / Mm_per_ds
        if base_radius <= 0:
            raise ValueError("base_radius must be positive.")
        self.radial_power = radial_power
        self.coordinate_radial_power = coordinate_radial_power
        self.base_radius = base_radius
        self.eps = eps

    def forward(self, coords, compute_jacobian=True):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.eps)
        normalized_radius = radius / self.base_radius
        coordinate_scale = normalized_radius.pow(-self.coordinate_radial_power)
        network_coords = coords * coordinate_scale
        a = SirenModel.forward(self, network_coords)
        a = a * normalized_radius.pow(-self.radial_power)
        b = curl(a, coords)

        out = {"b": b, "a": a, "network_coords": network_coords, "coordinate_scale": coordinate_scale}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


class SourceSurfaceHeightModel(SirenModel):
    """Bounded angular model for a warped spherical source-surface radius."""

    def __init__(self, height_range=(2.0, 2.5), initial_height=None,
                 hidden_dim=64, layers=3, w0=1.0, w0_initial=1.0, **kwargs):
        super().__init__(in_dim=3, out_dim=1, dim=hidden_dim, n_layers=layers,
                         w0=w0, w0_init=w0_initial, **kwargs)
        if len(height_range) != 2:
            raise ValueError("source_surface.height_range must contain [min, max].")
        min_height, max_height = [float(v) for v in height_range]
        if max_height <= min_height:
            raise ValueError("source_surface.height_range max must be greater than min.")
        initial_height = (min_height + max_height) / 2 if initial_height is None else float(initial_height)
        if not min_height < initial_height < max_height:
            raise ValueError("source_surface.initial_height must lie inside height_range.")

        self.register_buffer('height_min', torch.tensor(min_height, dtype=torch.float32))
        self.register_buffer('height_max', torch.tensor(max_height, dtype=torch.float32))

        initial_fraction = (initial_height - min_height) / (max_height - min_height)
        initial_fraction = min(max(initial_fraction, 1e-4), 1 - 1e-4)
        initial_logit = np.log(initial_fraction / (1 - initial_fraction))
        with torch.no_grad():
            self.out_layer.weight.zero_()
            self.out_layer.bias.fill_(initial_logit)

    def forward(self, unit_vectors):
        x = SirenModel.forward(self, unit_vectors)
        height_fraction = torch.sigmoid(x)
        return self.height_min + height_fraction * (self.height_max - self.height_min)


class SourceSurfaceScaledVectorPotentialModel(nn.Module):
    """Scaled vector-potential model with a learned radial source-surface transition."""

    def __init__(self, vector_potential=None, source_surface=None,
                 radial_power=2.0, coordinate_radial_power=4.0, radial_falloff_power=2.0, base_radius=None,
                 Mm_per_ds=None, eps=1e-6):
        super().__init__()

        # Model-unit geometry and shared radial scaling.
        if Mm_per_ds is None:
            raise ValueError("SourceSurfaceScaledVectorPotentialModel requires 'Mm_per_ds'.")
        if base_radius is None:
            base_radius = SOLAR_RADIUS_Mm / Mm_per_ds
        if base_radius <= 0:
            raise ValueError("base_radius must be positive.")
        self.radial_power = radial_power
        self.coordinate_radial_power = coordinate_radial_power
        self.radial_falloff_power = radial_falloff_power
        self.base_radius = base_radius
        self.eps = eps
        self.solar_radius_per_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        self.requires_grad_forward = True

        # Independent SIREN configs for the field and source-surface geometry.
        vector_potential = self._siren_config(vector_potential, default_hidden_dim=512, default_layers=8)
        source_surface = {} if source_surface is None else dict(source_surface)

        # Source-surface radii are configured in solar radii and converted once.
        source_surface_transition_width = float(source_surface.pop('transition_width', 0.05))
        if source_surface_transition_width <= 0:
            raise ValueError("source_surface.transition_width must be positive.")
        self.source_surface_eps = float(source_surface.pop('eps', 1e-6))
        self.source_surface_transition_width = source_surface_transition_width * self.solar_radius_per_ds
        source_surface.setdefault('height_range', [2.0, 2.5])
        self._convert_source_surface_config_units(source_surface)

        # Trainable submodels.
        self.vector_potential = SirenModel(in_dim=3, out_dim=3, **vector_potential)
        self.source_surface = SourceSurfaceHeightModel(**source_surface)

    def _convert_source_surface_config_units(self, source_surface):
        if 'height_range' in source_surface:
            source_surface['height_range'] = [
                float(v) * self.solar_radius_per_ds for v in source_surface['height_range']
            ]
        if 'initial_height' in source_surface and source_surface['initial_height'] is not None:
            source_surface['initial_height'] = float(source_surface['initial_height']) * self.solar_radius_per_ds

    @staticmethod
    def _siren_config(config, default_hidden_dim, default_layers):
        config = {} if config is None else dict(config)
        network_type = config.pop('type', 'siren')
        if network_type != 'siren':
            raise ValueError("Only SIREN source-surface submodels are supported.")
        hidden_dim = config.pop('hidden_dim', config.pop('dim', default_hidden_dim))
        layers = config.pop('layers', config.pop('n_layers', default_layers))
        w0 = config.pop('w0', 1.0)
        w0_initial = config.pop('w0_initial', config.pop('w0_init', 1.0))
        if config:
            keys = ', '.join(sorted(config))
            raise ValueError(f"Unsupported source-surface submodel config keys: {keys}")
        return {'dim': hidden_dim, 'n_layers': layers, 'w0': w0, 'w0_init': w0_initial}

    def _scaled_vector_potential_field(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.eps)
        normalized_radius = radius / self.base_radius
        coordinate_scale = normalized_radius.pow(-self.coordinate_radial_power)
        network_coords = coords * coordinate_scale
        a = self.vector_potential(network_coords)
        a = a * normalized_radius.pow(-self.radial_power)
        b = curl(a, coords)
        return b, a, network_coords, coordinate_scale

    def _source_surface_radius(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.source_surface_eps)
        unit_vectors = coords / radius
        ss_radius_model = self.source_surface(unit_vectors)
        return radius, ss_radius_model, unit_vectors

    def _source_surface_radial_field(self, radius, ss_radius_model, unit_vectors):
        ss_coords = unit_vectors * ss_radius_model
        b_surface, _, _, _ = self._scaled_vector_potential_field(ss_coords)
        br_surface = (b_surface * unit_vectors).sum(-1, keepdim=True)
        radial_decay = (ss_radius_model / radius).clamp_min(self.source_surface_eps).pow(self.radial_falloff_power)
        return br_surface * radial_decay * unit_vectors

    def forward(self, coords, compute_jacobian=True):
        # Vector-potential field at the requested coordinates.
        b_vector, a, network_coords, coordinate_scale = self._scaled_vector_potential_field(coords)

        # Learned warped source-surface radius in model units.
        radius, ss_radius_model, unit_vectors = self._source_surface_radius(coords)

        # Radial exterior: sample Br at the learned source surface and expand as r^-2 by default.
        b_radial = self._source_surface_radial_field(radius, ss_radius_model, unit_vectors)

        # Smoothly transition from the domain field to the source-surface-projected radial exterior.
        transition = torch.sigmoid(
            (radius - ss_radius_model) / self.source_surface_transition_width)
        b = (1 - transition) * b_vector + transition * b_radial

        out = {
            "b": b,
            "a": a,
            "b_vector_potential": b_vector,
            "b_radial": b_radial,
            "source_surface_weight": transition,
            "source_surface_radius": ss_radius_model / self.solar_radius_per_ds,
            "source_surface_radius_model": ss_radius_model,
            "network_coords": network_coords,
            "coordinate_scale": coordinate_scale,
        }
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out

    @torch.no_grad()
    def source_surface_height_grid(self, latitude_resolution=180, longitude_resolution=360, device=None):
        device = next(self.parameters()).device if device is None else device
        latitude = torch.linspace(-np.pi / 2, np.pi / 2, int(latitude_resolution), device=device)
        longitude = torch.linspace(0, 2 * np.pi, int(longitude_resolution), device=device)
        lat_grid, lon_grid = torch.meshgrid(latitude, longitude, indexing='ij')
        unit_vectors = torch.stack([
            torch.cos(lat_grid) * torch.cos(lon_grid),
            torch.cos(lat_grid) * torch.sin(lon_grid),
            torch.sin(lat_grid),
        ], dim=-1)
        radius_model = self.source_surface(unit_vectors.reshape(-1, 3)).reshape(lat_grid.shape)
        return {
            'radius': radius_model / self.solar_radius_per_ds,
            'latitude': latitude,
            'longitude': longitude,
        }


class SourceSurfaceScaledPotentialModel(nn.Module):
    """Scaled scalar-potential model with a learned radial source-surface transition."""

    def __init__(self, potential=None, source_surface=None,
                 radial_power=1.0, coordinate_radial_power=4.0, radial_falloff_power=2.0, base_radius=None,
                 Mm_per_ds=None, eps=1e-6):
        super().__init__()

        if Mm_per_ds is None:
            raise ValueError("SourceSurfaceScaledPotentialModel requires 'Mm_per_ds'.")
        if base_radius is None:
            base_radius = SOLAR_RADIUS_Mm / Mm_per_ds
        if base_radius <= 0:
            raise ValueError("base_radius must be positive.")
        self.radial_power = radial_power
        self.coordinate_radial_power = coordinate_radial_power
        self.radial_falloff_power = radial_falloff_power
        self.base_radius = base_radius
        self.eps = eps
        self.solar_radius_per_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        self.requires_grad_forward = True

        potential = SourceSurfaceScaledVectorPotentialModel._siren_config(
            potential, default_hidden_dim=512, default_layers=8)
        source_surface = {} if source_surface is None else dict(source_surface)

        source_surface_transition_width = float(source_surface.pop('transition_width', 0.05))
        if source_surface_transition_width <= 0:
            raise ValueError("source_surface.transition_width must be positive.")
        self.source_surface_eps = float(source_surface.pop('eps', 1e-6))
        self.source_surface_transition_width = source_surface_transition_width * self.solar_radius_per_ds
        source_surface.setdefault('height_range', [2.0, 2.5])
        self._convert_source_surface_config_units(source_surface)

        self.potential = SirenModel(in_dim=3, out_dim=1, **potential)
        self.source_surface = SourceSurfaceHeightModel(**source_surface)

    def _convert_source_surface_config_units(self, source_surface):
        if 'height_range' in source_surface:
            source_surface['height_range'] = [
                float(v) * self.solar_radius_per_ds for v in source_surface['height_range']
            ]
        if 'initial_height' in source_surface and source_surface['initial_height'] is not None:
            source_surface['initial_height'] = float(source_surface['initial_height']) * self.solar_radius_per_ds

    def _scaled_potential_field(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.eps)
        normalized_radius = radius / self.base_radius
        coordinate_scale = normalized_radius.pow(-self.coordinate_radial_power)
        network_coords = coords * coordinate_scale
        phi = self.potential(network_coords)
        phi = phi * normalized_radius.pow(-self.radial_power)
        b = -gradient(phi, coords)
        return b, phi, network_coords, coordinate_scale

    def _source_surface_radius(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.source_surface_eps)
        unit_vectors = coords / radius
        ss_radius_model = self.source_surface(unit_vectors)
        return radius, ss_radius_model, unit_vectors

    def _source_surface_radial_field(self, radius, ss_radius_model, unit_vectors):
        ss_coords = unit_vectors * ss_radius_model
        b_surface, _, _, _ = self._scaled_potential_field(ss_coords)
        br_surface = (b_surface * unit_vectors).sum(-1, keepdim=True)
        radial_decay = (ss_radius_model / radius).clamp_min(self.source_surface_eps).pow(self.radial_falloff_power)
        return br_surface * radial_decay * unit_vectors

    def forward(self, coords, compute_jacobian=True):
        b_potential, phi, network_coords, coordinate_scale = self._scaled_potential_field(coords)

        radius, ss_radius_model, unit_vectors = self._source_surface_radius(coords)
        b_radial = self._source_surface_radial_field(radius, ss_radius_model, unit_vectors)

        transition = torch.sigmoid(
            (radius - ss_radius_model) / self.source_surface_transition_width)
        b = (1 - transition) * b_potential + transition * b_radial

        out = {
            "b": b,
            "phi": phi,
            "b_potential": b_potential,
            "b_radial": b_radial,
            "source_surface_weight": transition,
            "source_surface_radius": ss_radius_model / self.solar_radius_per_ds,
            "source_surface_radius_model": ss_radius_model,
            "network_coords": network_coords,
            "coordinate_scale": coordinate_scale,
        }
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out

    @torch.no_grad()
    def source_surface_height_grid(self, latitude_resolution=180, longitude_resolution=360, device=None):
        device = next(self.parameters()).device if device is None else device
        latitude = torch.linspace(-np.pi / 2, np.pi / 2, int(latitude_resolution), device=device)
        longitude = torch.linspace(0, 2 * np.pi, int(longitude_resolution), device=device)
        lat_grid, lon_grid = torch.meshgrid(latitude, longitude, indexing='ij')
        unit_vectors = torch.stack([
            torch.cos(lat_grid) * torch.cos(lon_grid),
            torch.cos(lat_grid) * torch.sin(lon_grid),
            torch.sin(lat_grid),
        ], dim=-1)
        radius_model = self.source_surface(unit_vectors.reshape(-1, 3)).reshape(lat_grid.shape)
        return {
            'radius': radius_model / self.solar_radius_per_ds,
            'latitude': latitude,
            'longitude': longitude,
        }


def calculate_current(b, coords, jac_matrix=None):
    jac_matrix = jacobian(b, coords) if jac_matrix is None else jac_matrix
    return calculate_current_from_jacobian(jac_matrix)


def calculate_current_from_jacobian(jac_matrix, f=torch):
    dBy_dx = jac_matrix[..., 1, 0]
    dBz_dx = jac_matrix[..., 2, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBz_dy = jac_matrix[..., 2, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]

    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    return f.stack([rot_x, rot_y, rot_z], -1)


def curl(vector, coords):
    jac_matrix = jacobian(vector, coords)
    dVy_dx = jac_matrix[:, 1, 0]
    dVz_dx = jac_matrix[:, 2, 0]
    dVx_dy = jac_matrix[:, 0, 1]
    dVz_dy = jac_matrix[:, 2, 1]
    dVx_dz = jac_matrix[:, 0, 2]
    dVy_dz = jac_matrix[:, 1, 2]
    rot_x = dVz_dy - dVy_dz
    rot_y = dVx_dz - dVz_dx
    rot_z = dVy_dx - dVx_dy
    return torch.stack([rot_x, rot_y, rot_z], -1)


def gradient(scalar, coords):
    return torch.autograd.grad(
        scalar[:, 0],
        coords,
        grad_outputs=torch.ones_like(scalar[:, 0]).to(scalar),
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]


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
