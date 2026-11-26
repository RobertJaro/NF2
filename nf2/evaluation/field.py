"""Shared checkpoint and validation field evaluation in normalized coordinates."""
import torch
from torch import nn

from nf2.train.model import FreeBoundaryVectorPotentialModel, jacobian


def evaluate_field(model, coords, compute_jacobian=True, radial_extension=True):
    if not radial_extension or not isinstance(model, FreeBoundaryVectorPotentialModel):
        return model(coords, compute_jacobian=compute_jacobian)

    radius = coords.norm(dim=-1, keepdim=True).clamp_min(model.eps)
    direction = coords / radius
    surface = model.surface_model(direction)
    inside = model.inside_source_surface(radius, surface)
    sample = torch.where(inside, coords, direction * surface)
    result = model(sample, compute_jacobian=False)
    b = result["b"]
    exterior_b = (b * direction).sum(dim=-1, keepdim=True) * direction * (surface / radius).square()
    result["b"] = torch.where(inside, b, exterior_b)
    result["inside_source_surface"] = inside
    result.pop("a", None)
    if compute_jacobian:
        result["jac_matrix"] = jacobian(result["b"], coords)
    return result


class FieldEvaluator(nn.Module):
    """Expose field evaluation as a module for DataParallel."""

    def __init__(self, model, radial_extension=True):
        super().__init__()
        self.model = model
        self.radial_extension = radial_extension

    def forward(self, coords, compute_jacobian=True):
        return evaluate_field(self.model, coords, compute_jacobian, self.radial_extension)
