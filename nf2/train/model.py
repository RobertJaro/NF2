import numpy as np
import torch
from torch import nn
from torch.func import functional_call
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

    def vector_potential(self, coords):
        return {"a": SirenModel.forward(self, coords)}

    def forward(self, coords, compute_jacobian=True):
        out = self.vector_potential(coords)
        a = out["a"]
        b = curl(a, coords)

        out["b"] = b
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


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

    def vector_potential(self, coords):
        radius = coords.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(self.eps)
        normalized_radius = radius / self.base_radius
        coordinate_scale = normalized_radius.pow(-self.coordinate_radial_power)
        network_coords = coords * coordinate_scale
        a = SirenModel.forward(self, network_coords)
        a = a * normalized_radius.pow(-self.radial_power)
        return {"a": a, "network_coords": network_coords, "coordinate_scale": coordinate_scale}

    def forward(self, coords, compute_jacobian=True):
        out = self.vector_potential(coords)
        a = out["a"]
        b = curl(a, coords)

        out["b"] = b
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


class SourceSurfaceHeightModel(nn.Module):
    """Map unit directions to a bounded source-surface radius."""

    def __init__(self, height_range=(1.5, 2.5), Mm_per_ds=None,
                 hidden_dim=64, layers=3, w0=1.0, w0_initial=1.0):
        super().__init__()
        if Mm_per_ds is None:
            raise ValueError("SourceSurfaceHeightModel requires 'Mm_per_ds'.")
        if len(height_range) != 2:
            raise ValueError("Source-surface height_range must contain [min, max].")
        min_height, max_height = [float(v) for v in height_range]
        if max_height <= min_height:
            raise ValueError("Source-surface height_range max must be greater than min.")
        self.mapping_module = SirenModel(
            in_dim=3, out_dim=1, dim=hidden_dim, n_layers=layers, w0=w0, w0_init=w0_initial)
        self.solar_radius_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        self.register_buffer('height_min', torch.tensor(min_height * self.solar_radius_ds, dtype=torch.float32))
        self.register_buffer('height_max', torch.tensor(max_height * self.solar_radius_ds, dtype=torch.float32))

    def forward(self, unit_vectors, freeze_parameters=False):
        if freeze_parameters:
            state = {
                name: parameter.detach()
                for name, parameter in self.mapping_module.named_parameters()
            }
            state.update(dict(self.mapping_module.named_buffers()))
            height_logits = functional_call(self.mapping_module, state, (unit_vectors,))
        else:
            height_logits = self.mapping_module(unit_vectors)

        height_fraction = torch.sigmoid(height_logits)
        return self.height_min + height_fraction * (self.height_max - self.height_min)


class SourceSurfaceOpenFieldModel(nn.Module):
    """Map unit directions to a tangential angular vector potential."""

    def __init__(self, hidden_dim=64, layers=3, w0=1.0, w0_initial=1.0):
        super().__init__()
        self.mapping_module = SirenModel(
            in_dim=3, out_dim=3, dim=hidden_dim, n_layers=layers, w0=w0, w0_init=w0_initial)

    def forward(self, unit_vectors):
        angular_potential = self.mapping_module(unit_vectors)
        radial_component = (angular_potential * unit_vectors).sum(dim=-1, keepdim=True)
        return angular_potential - radial_component * unit_vectors


class OpenVectorPotentialModel(nn.Module):
    """Add an unscaled interior vector potential to a radial open-field potential."""

    requires_grad_forward = True
    interior_model_class = VectorPotentialModel

    def __init__(self, open_field=None, eps=1e-6, **field_kwargs):
        super().__init__()
        self.field_model = self.interior_model_class(**field_kwargs)
        self.open_field_model = SourceSurfaceOpenFieldModel(**(open_field or {}))
        self.eps = float(eps)

    def forward(self, coords, compute_jacobian=True):
        coords.requires_grad_(True)

        radius = coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        unit_vectors = coords / radius

        interior_a = self.field_model.vector_potential(coords)["a"]
        open_a = self.open_field_model(unit_vectors) / radius
        a = interior_a + open_a
        b = curl(a, coords)

        out = {"a": a, "b": b}
        if compute_jacobian:
            out["jac_matrix"] = jacobian(b, coords)
        return out


class OpenScaledVectorPotentialModel(OpenVectorPotentialModel):
    """Add a decaying interior vector potential to a radial open-field potential."""

    interior_model_class = ScaledVectorPotentialModel

    def __init__(self, radial_power=2.0, coordinate_radial_power=0.0,
                 Mm_per_ds=None, **kwargs):
        if Mm_per_ds is None:
            raise ValueError("OpenScaledVectorPotentialModel requires 'Mm_per_ds'.")
        super().__init__(
            radial_power=radial_power,
            coordinate_radial_power=coordinate_radial_power,
            Mm_per_ds=Mm_per_ds,
            **kwargs,
        )


class SourceSurfaceVectorPotentialModel(nn.Module):
    """Blend an interior potential into an open potential across a learned surface."""

    requires_grad_forward = True

    def __init__(self, source_surface=None, open_field=None, Mm_per_ds=None, eps=1e-6, **field_kwargs):
        super().__init__()
        if Mm_per_ds is None:
            raise ValueError("SourceSurfaceVectorPotentialModel requires 'Mm_per_ds'.")

        source_surface = dict(source_surface or {})
        transition_width = float(source_surface.pop('transition_width', 0.1))
        if transition_width <= 0:
            raise ValueError('source_surface.transition_width must be positive.')

        self.field_model = ScaledVectorPotentialModel(Mm_per_ds=Mm_per_ds, eps=eps, **field_kwargs)
        self.open_field_model = SourceSurfaceOpenFieldModel(**(open_field or {}))
        self.surface_model = SourceSurfaceHeightModel(Mm_per_ds=Mm_per_ds, **source_surface)

        self.solar_radius_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        self.transition_width_ds = transition_width * self.solar_radius_ds
        self.eps = float(eps)

    def source_surface_gate(self, radial_distance_ds):
        return torch.sigmoid(radial_distance_ds / self.transition_width_ds)

    def forward(self, coords, compute_jacobian=True):
        coords.requires_grad_(True)

        radius_ds = coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        unit_vectors = coords / radius_ds

        source_radius_ds = self.surface_model(unit_vectors)
        transition = self.source_surface_gate(source_radius_ds - radius_ds)

        fixed_source_radius_ds = self.surface_model(unit_vectors, freeze_parameters=True)
        field_gate = self.source_surface_gate(fixed_source_radius_ds - radius_ds)

        interior_a = self.field_model.vector_potential(coords)['a']
        open_a = self.open_field_model(unit_vectors) / radius_ds
        a = field_gate * interior_a + open_a
        b = curl(a, coords)

        out = {
            'a': a,
            'b': b,
            'source_surface_gate': field_gate.detach(),
            'source_surface_radius': source_radius_ds / self.solar_radius_ds,
            'source_surface_transition': transition,
            'inside_source_surface': radius_ds <= source_radius_ds,
        }
        if compute_jacobian:
            out['jac_matrix'] = jacobian(b, coords)
        return out


class FreeBoundaryVectorPotentialModel(ScaledVectorPotentialModel):
    """Scaled potential inside a learned domain, with a separately penalized radial boundary.

    Field derivatives are always taken in physical Cartesian model coordinates.
    The surface affects training through sample locations, not through a field gate.
    Both interior and boundary sample locations propagate gradients to the surface.
    """

    def __init__(self, source_surface=None, Mm_per_ds=None, **kwargs):
        super().__init__(Mm_per_ds=Mm_per_ds, **kwargs)
        surface = dict(source_surface or {})
        height_range = surface.get('height_range', (1.5, 2.5))
        if len(height_range) != 2 or not 1 < height_range[0] < height_range[1]:
            raise ValueError('Free boundary height_range must satisfy 1 < min < max.')
        initial_radius = float(surface.pop('initial_radius', sum(height_range) / 2))
        if not height_range[0] < initial_radius < height_range[1]:
            raise ValueError('initial_radius must be strictly inside height_range.')
        self.surface_model = SourceSurfaceHeightModel(Mm_per_ds=Mm_per_ds, **surface)
        self.solar_radius_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        fraction = (initial_radius - height_range[0]) / (height_range[1] - height_range[0])
        # Begin with a sphere; angular structure develops during optimization.
        nn.init.zeros_(self.surface_model.mapping_module.out_layer.weight)
        nn.init.constant_(self.surface_model.mapping_module.out_layer.bias,
                          float(np.log(fraction / (1 - fraction))))

    def map_samples(self, batch):
        """Map reference directions and fractions to r = 1 + u (Rss - 1)."""
        reference = batch['coords']
        directions = (reference / reference.norm(dim=-1, keepdim=True)).detach()
        source_radius = self.surface_model(directions) / self.solar_radius_ds
        fraction = batch['source_surface_fraction']
        # Interior physics and the radial boundary loss both update Rss through
        # sample locations; the separate height penalty opposes expansion.
        radius = 1 + fraction * (source_radius - 1)
        coords = directions * radius * self.solar_radius_ds
        out = {'coords': coords, 'source_surface_radius': source_radius}
        if 'grouped_coords' in batch:
            out['grouped_coords'] = coords.reshape_as(batch['grouped_coords'])
        if 'spherical_coords' in batch:
            out['spherical_coords'] = torch.cat([radius, batch['spherical_coords'][:, 1:]], dim=-1)
        return out

    @staticmethod
    def inside_source_surface(radius, source_radius):
        # Recovering a radius from Cartesian boundary coordinates introduces roundoff.
        return radius <= source_radius * (1 + 1e-6)

    def forward(self, coords, compute_jacobian=True):
        out = super().forward(coords, compute_jacobian=compute_jacobian)
        radius = coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        source_radius = self.surface_model(coords / radius)
        out['source_surface_radius'] = source_radius / self.solar_radius_ds
        out['inside_source_surface'] = self.inside_source_surface(radius, source_radius)
        return out


class FixedSourceSurfaceVectorPotentialModel(nn.Module):
    """Combine an interior field with a radial open-field background across a fixed sphere."""

    requires_grad_forward = True

    def __init__(self, source_surface=None, open_field=None, Mm_per_ds=None, eps=1e-6, **field_kwargs):
        super().__init__()
        if Mm_per_ds is None:
            raise ValueError("FixedSourceSurfaceVectorPotentialModel requires 'Mm_per_ds'.")
        source_surface = dict(source_surface or {})
        height = float(source_surface.pop('height', 2.0))
        transition_width = float(source_surface.pop('transition_width', 0.1))
        if source_surface:
            raise ValueError(
                f"Unsupported fixed source-surface options: {', '.join(sorted(source_surface))}.")
        if height <= 0:
            raise ValueError('source_surface.height must be positive.')
        if transition_width <= 0:
            raise ValueError('source_surface.transition_width must be positive.')

        self.field_model = ScaledVectorPotentialModel(Mm_per_ds=Mm_per_ds, eps=eps, **field_kwargs)
        self.open_field_model = SourceSurfaceOpenFieldModel(**(open_field or {}))

        self.solar_radius_ds = SOLAR_RADIUS_Mm / Mm_per_ds
        self.register_buffer(
            'source_surface_radius_ds',
            torch.tensor(height * self.solar_radius_ds, dtype=torch.float32),
        )
        self.transition_width_ds = transition_width * self.solar_radius_ds
        self.eps = float(eps)

    def source_surface_gate(self, radial_distance_ds):
        return torch.sigmoid(radial_distance_ds / self.transition_width_ds)

    def forward(self, coords, compute_jacobian=True):
        coords.requires_grad_(True)

        radius_ds = coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        unit_vectors = coords / radius_ds
        radial_distance_ds = self.source_surface_radius_ds - radius_ds
        gate = self.source_surface_gate(radial_distance_ds)

        interior_a = self.field_model.vector_potential(coords)['a']
        open_a = self.open_field_model(unit_vectors) / radius_ds
        a = open_a + gate * interior_a
        b = curl(a, coords)

        out = {
            'a': a,
            'b': b,
            'source_surface_gate': gate,
            'source_surface_radius': torch.ones_like(radius_ds)
                                     * (self.source_surface_radius_ds / self.solar_radius_ds),
            'inside_source_surface': radial_distance_ds >= 0,
        }
        if compute_jacobian:
            out['jac_matrix'] = jacobian(b, coords)
        return out


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
