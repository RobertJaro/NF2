import torch

import nf2.train.model as model_module
from nf2.train.model import (
    OpenScaledVectorPotentialModel, OpenVectorPotentialModel,
    ScaledVectorPotentialModel, VectorPotentialModel,
)


def _model():
    return OpenVectorPotentialModel(
        dim=8,
        n_layers=1,
        open_field={"hidden_dim": 8, "layers": 1},
    )


def test_open_vector_potential_combines_unscaled_potentials_before_one_curl(monkeypatch):
    model = _model()
    coords = torch.tensor([[1.2, 0.0, 0.0], [0.0, 1.6, 0.0]], requires_grad=True)

    curl_calls = 0
    original_curl = model_module.curl

    def counted_curl(*args, **kwargs):
        nonlocal curl_calls
        curl_calls += 1
        return original_curl(*args, **kwargs)

    monkeypatch.setattr(model_module, "curl", counted_curl)
    out = model(coords, compute_jacobian=False)

    radius = coords.norm(dim=-1, keepdim=True)
    unit_vectors = coords / radius
    interior_a = model.field_model.vector_potential(coords)["a"]
    open_a = model.open_field_model(unit_vectors) / radius

    assert isinstance(model.field_model, VectorPotentialModel)
    assert not isinstance(model.field_model, ScaledVectorPotentialModel)
    assert curl_calls == 1
    torch.testing.assert_close(out["a"], interior_a + open_a)


def test_open_vector_potential_open_component_is_radial():
    torch.manual_seed(0)
    model = _model()
    unit_vectors = torch.randn(32, 3)
    unit_vectors = unit_vectors / unit_vectors.norm(dim=-1, keepdim=True)
    coords = (2.0 * unit_vectors).requires_grad_()

    radius = coords.norm(dim=-1, keepdim=True)
    open_a = model.open_field_model(coords / radius) / radius
    open_b = model_module.curl(open_a, coords)

    torch.testing.assert_close(
        torch.cross(open_b, unit_vectors, dim=-1),
        torch.zeros_like(open_b),
        atol=2e-5,
        rtol=1e-5,
    )


def test_open_vector_potential_total_field_is_divergence_free():
    torch.manual_seed(0)
    model = _model()
    coords = torch.randn(16, 3)
    coords = coords / coords.norm(dim=-1, keepdim=True) * 1.5

    out = model(coords, compute_jacobian=True)
    divergence = out["jac_matrix"].diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    torch.testing.assert_close(
        divergence,
        torch.zeros_like(divergence),
        atol=2e-5,
        rtol=1e-5,
    )


def test_open_scaled_vector_potential_uses_only_amplitude_scaling_by_default():
    model = OpenScaledVectorPotentialModel(
        Mm_per_ds=695.7,
        dim=8,
        n_layers=1,
        open_field={"hidden_dim": 8, "layers": 1},
    )

    assert isinstance(model.field_model, ScaledVectorPotentialModel)
    assert model.field_model.radial_power == 2.0
    assert model.field_model.coordinate_radial_power == 0.0

    coords = torch.tensor([[1.5, 0.0, 0.0]], requires_grad=True)
    out = model(coords, compute_jacobian=False)
    assert out["a"].shape == (1, 3)
    assert out["b"].shape == (1, 3)
