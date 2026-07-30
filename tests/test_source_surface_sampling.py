from types import SimpleNamespace

import numpy as np
import torch

import nf2
import nf2.train.callback as callback_module
from nf2.evaluation.output import BaseOutput
from nf2.train.callback import MetricsCallback, SourceSurfaceCallback, SphericalSlicesCallback
from nf2.train.loss import ForceFreeLoss, SourceSurfaceTransitionRadialLoss, loss_module_mapping
from nf2.train.model import (
    FixedSourceSurfaceVectorPotentialModel, SourceSurfaceVectorPotentialModel, curl,
)
from nf2.train.module import NF2Module


def _source_surface_model():
    return SourceSurfaceVectorPotentialModel(
        dim=8, n_layers=1, base_radius=1.0, Mm_per_ds=695.7,
        source_surface={"height_range": [1.5, 1.7], "transition_width": 0.1},
        open_field={"hidden_dim": 8, "layers": 1},
    )


def _fixed_source_surface_model():
    return FixedSourceSurfaceVectorPotentialModel(
        dim=8, n_layers=1, base_radius=1.0, Mm_per_ds=695.7,
        source_surface={"height": 2.0, "transition_width": 0.1},
    )


def test_learned_source_surface_uses_live_and_fixed_parameter_height_passes(monkeypatch):
    model = _source_surface_model()
    calls = 0
    original_forward = model.surface_model.forward

    def counted_forward(unit_vectors, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(unit_vectors, **kwargs)

    monkeypatch.setattr(model.surface_model, "forward", counted_forward)
    result = model(torch.eye(3) * 1.6, compute_jacobian=False)

    assert calls == 2
    assert model.field_model is not model.open_field_model
    assert model.surface_model is not model.open_field_model
    assert torch.all((result["source_surface_radius"] >= 1.5)
                     & (result["source_surface_radius"] <= 1.7))


def test_learned_source_surface_uses_complementary_potential_gate():
    model = _source_surface_model()
    coords = torch.tensor([[1.2, 0.0, 0.0], [1.8, 0.0, 0.0]], requires_grad=True)

    out = model(coords, compute_jacobian=False)
    radius = coords.norm(dim=-1, keepdim=True)
    unit_vectors = coords / radius
    source_radius = model.surface_model(unit_vectors)
    transition = torch.sigmoid((source_radius - radius) / model.transition_width_ds)
    interior_a = model.field_model.vector_potential(coords)["a"]
    open_a = model.open_field_model(unit_vectors) / radius

    torch.testing.assert_close(out["a"], transition * interior_a + (1 - transition) * open_a)
    torch.testing.assert_close(out["source_surface_transition"], transition)
    torch.testing.assert_close(out["source_surface_gate"], transition.detach())
    assert not out["source_surface_gate"].requires_grad


def test_learned_source_surface_open_component_is_radial():
    model = _source_surface_model()
    coords = torch.tensor([[5.0, 0.0, 0.0]], requires_grad=True)
    radius = coords.norm(dim=-1, keepdim=True)
    unit_vectors = coords / coords.norm(dim=-1, keepdim=True)
    open_a = model.open_field_model(unit_vectors) / radius
    open_b = curl(open_a, coords)

    torch.testing.assert_close(
        torch.cross(open_b, unit_vectors, dim=-1),
        torch.zeros_like(open_b),
        atol=2e-5,
        rtol=1e-5,
    )


def test_learned_source_surface_field_is_divergence_free():
    torch.manual_seed(0)
    model = _source_surface_model()
    coords = torch.randn(16, 3)
    coords = coords / coords.norm(dim=-1, keepdim=True) * 1.6

    out = model(coords, compute_jacobian=True)
    divergence = out["jac_matrix"].diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=2e-5, rtol=1e-5)


def test_fixed_source_surface_model_uses_physical_coordinates_and_returns_gate():
    model = _fixed_source_surface_model()
    coords = torch.tensor([[1.9, 0.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0]])

    out = model(coords, compute_jacobian=False)

    torch.testing.assert_close(
        out["source_surface_gate"],
        torch.sigmoid(torch.tensor([[1.0], [0.0], [-1.0]])),
    )
    torch.testing.assert_close(out["source_surface_radius"], torch.full((3, 1), 2.0))
    assert out["inside_source_surface"].flatten().tolist() == [True, True, False]


def test_force_free_loss_uses_normalized_source_surface_gate():
    loss = ForceFreeLoss(name="force_free", ds_id="random")
    b = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    jac_matrix = torch.zeros(2, 3, 3)
    jac_matrix[:, 0, 2] = torch.tensor([1.0, 2.0])
    gate = torch.tensor([[0.25], [0.75]])

    unweighted = loss(b=b, jac_matrix=jac_matrix, coords=torch.zeros(2, 3))
    weighted = loss(
        b=b, jac_matrix=jac_matrix, coords=torch.zeros(2, 3),
        source_surface_gate=gate,
    )

    torch.testing.assert_close(weighted, unweighted * gate.squeeze(-1) / gate.mean())
    assert not torch.equal(unweighted, weighted)


def test_source_surface_transition_radial_loss_uses_differentiable_shell_weights():
    loss = SourceSurfaceTransitionRadialLoss(
        name="source_surface_transition_radial", ds_id="random")
    coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)
    transition = torch.tensor([[0.2], [0.6]], requires_grad=True)

    value = loss(b=b, coords=coords, source_surface_transition=transition)
    value.backward()

    torch.testing.assert_close(value, torch.tensor(0.6))
    assert transition.grad is not None
    assert b.grad is not None
    assert loss_module_mapping[
        "source_surface_transition_radial"] is SourceSurfaceTransitionRadialLoss


def test_force_free_loss_does_not_update_learned_surface():
    model = _source_surface_model()
    coords = torch.tensor([
        [1.2, 0.0, 0.0],
        [0.0, 1.6, 0.0],
    ])
    out = model(coords, compute_jacobian=True)

    ForceFreeLoss(name="force_free", ds_id="random")(
        coords=coords, **out).mean().backward()

    assert all(parameter.grad is None for parameter in model.surface_model.parameters())


def test_checkpoint_output_uses_complete_learned_source_surface_model():
    model = _source_surface_model()
    output = BaseOutput({
        "model": model,
        "transforms": [],
        "data": {"Mm_per_ds": 695.7, "Gauss_per_dB": 1000},
    }, device=torch.device("cpu"))

    sampled = output._sample_tensor(torch.tensor([[2.0, 0.0, 0.0]]), compute_jacobian=False)

    assert "source_surface_transition" in sampled
    assert not sampled["inside_source_surface"].item()


def test_public_loader_restores_complete_source_surface_model(tmp_path):
    checkpoint_path = tmp_path / "source_surface.nf2"
    torch.save({
        "model": _source_surface_model(),
        "transforms": [],
        "data": {
            "type": "spherical",
            "radius_range": [1.0, 2.5],
            "Mm_per_ds": 695.7,
            "Gauss_per_dB": 1000,
        },
    }, checkpoint_path)

    output = nf2.load(checkpoint_path, device=torch.device("cpu"))
    sampled = output._sample_tensor(torch.tensor([[2.0, 0.0, 0.0]]), compute_jacobian=False)
    loaded_model = output.model.module if hasattr(output.model, "module") else output.model

    assert isinstance(loaded_model, SourceSurfaceVectorPotentialModel)
    assert "source_surface_transition" in sampled


def test_validation_uses_combined_source_surface_field():
    module = NF2Module(
        validation_mapping={0: "sphere"},
        data_config={"Mm_per_ds": 695.7, "Gauss_per_dB": 1000},
        model_kwargs={
            "type": "source_surface_vector_potential",
            "dim": 8,
            "n_layers": 1,
            "base_radius": 1.0,
            "source_surface": {"height_range": [1.5, 1.7], "transition_width": 0.1},
            "open_field": {"hidden_dim": 8, "layers": 1},
        },
        loss_config=[],
        lr_params=1e-3,
    )

    output = module.validation_step({"coords": torch.tensor([
        [1.2, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])}, 0, dataloader_idx=0)

    assert output["inside_source_surface"].flatten().tolist() == [True, False]
    assert output["jac_matrix"].shape == (2, 3, 3)


def test_full_volume_training_preserves_grouped_height_scaling():
    module = NF2Module(
        validation_mapping={},
        data_config={"Mm_per_ds": 695.7, "Gauss_per_dB": 1000},
        model_kwargs={
            "type": "source_surface_vector_potential",
            "dim": 8,
            "n_layers": 1,
            "base_radius": 1.0,
            "source_surface": {"height_range": [1.5, 1.7], "transition_width": 0.1},
            "open_field": {"hidden_dim": 8, "layers": 1},
        },
        loss_config=[
            {"type": "force_free", "name": "force_free", "weight": 1e-3, "ds_id": "random"},
            {"type": "source_surface_transition_radial",
             "name": "source_surface_transition_radial", "weight": 1e-2, "ds_id": "random"},
        ],
        transforms=[],
        loss_scaling=[{"type": "b_height", "loss_ids": ["force_free"]}],
        lr_params=1e-3,
    )
    grouped_coords = torch.tensor([
        [[1.1, 0.0, 0.0], [1.8, 0.0, 0.0]],
        [[0.0, 1.1, 0.0], [0.0, 1.8, 0.0]],
    ])
    batch = {
        "random": {
            "coords": grouped_coords.reshape(-1, 3),
            "grouped_coords": grouped_coords,
            "requires_jacobian": torch.tensor(True),
        },
    }

    output = module.training_step(batch, 0)
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert 1.5 <= output["source_surface_mean_radius"] <= 1.7
    surface_grads = [
        parameter.grad for parameter in module.model.surface_model.parameters()
        if parameter.grad is not None
    ]
    assert surface_grads
    assert all(torch.isfinite(gradient).all() for gradient in surface_grads)


def test_source_surface_callback_plots_deformed_radius_grid(monkeypatch):
    model = _source_surface_model()
    expected_radius = torch.tensor([[1.5, 1.6, 1.7], [1.7, 1.6, 1.5]])
    model.source_surface_radius_grid = lambda **kwargs: {
        "radius": expected_radius,
        "latitude": torch.tensor([-np.pi / 2, np.pi / 2]),
        "longitude": torch.tensor([0.0, np.pi, 2 * np.pi]),
    }
    scalar_logs = []
    plotted = []
    monkeypatch.setattr(callback_module.wandb, "log", lambda values: scalar_logs.append(values))
    monkeypatch.setattr(
        callback_module,
        "_log_wandb_figure",
        lambda name, fig: plotted.append((name, np.asarray(fig.axes[0].images[0].get_array()))),
    )

    SourceSurfaceCallback(resolution=(2, 3)).on_validation_end(
        SimpleNamespace(), SimpleNamespace(model=model, device=torch.device("cpu")))

    assert abs(scalar_logs[0]["source_surface/min_radius"] - 1.5) < 1e-6
    assert abs(scalar_logs[0]["source_surface/max_radius"] - 1.7) < 1e-6
    assert plotted[0][0] == "source_surface - Radius"
    np.testing.assert_allclose(plotted[0][1], expected_radius.numpy())


def test_metrics_callback_uses_only_points_below_source_surface(monkeypatch):
    logs = []
    monkeypatch.setattr(callback_module.wandb, "log", lambda values: logs.append(values))
    module = SimpleNamespace(validation_outputs={"sphere": {
        "b": torch.tensor([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
        "j": torch.tensor([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        "div": torch.tensor([0.5, 100.0]),
        "inside_source_surface": torch.tensor([[True], [False]]),
    }})

    MetricsCallback("sphere", gauss_per_dB=1.0, Mm_per_ds=1.0).on_validation_end(
        SimpleNamespace(), module)

    assert len(logs) == 1
    assert abs(float(logs[0]["valid"]["divergence"]) - 0.5) < 1e-5


def test_spherical_slice_callback_plots_surface_contour(monkeypatch):
    shape = (2, 3, 4)
    radius = torch.tensor([1.4, 1.8])[:, None, None].expand(shape)
    colatitude = torch.linspace(0.3, np.pi - 0.3, shape[1])[None, :, None].expand(shape)
    longitude = torch.linspace(0.0, 2 * np.pi, shape[2])[None, None, :].expand(shape)
    spherical_coords = torch.stack([radius, colatitude, longitude], dim=-1)
    inside = torch.tensor([
        [[True, True, False, False]] * shape[1],
        [[False, True, True, False]] * shape[1],
    ])
    outputs = {
        "b": torch.ones(*shape, 3).reshape(-1, 3),
        "j": torch.linspace(0.1, 1.0, int(np.prod(shape) * 3)).reshape(-1, 3),
        "coords": torch.zeros(int(np.prod(shape)), 3),
        "spherical_coords": spherical_coords.reshape(-1, 3),
        "inside_source_surface": inside.reshape(-1, 1),
    }
    plotted = []
    monkeypatch.setattr(
        callback_module, "_log_wandb_figure", lambda name, fig: plotted.append(name))

    SphericalSlicesCallback(
        "slices", shape, gauss_per_dB=1.0, Mm_per_ds=1.0).on_validation_end(
        SimpleNamespace(), SimpleNamespace(validation_outputs={"slices": outputs}))

    assert plotted == [
        "slices - B",
        "slices - Current density",
        "slices - Integrated Current density",
    ]
