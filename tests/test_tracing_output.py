from unittest.mock import patch

import numpy as np
import torch
from astropy import units as u

import nf2
from nf2.train.model import BModel


def _constant_checkpoint(path):
    model = BModel(dim=8, n_layers=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.out_layer.bias[2] = 1
    torch.save({
        "model": model,
        "data": {
            "type": "cartesian",
            "coord_range": np.array([[0, 1], [0, 1], [0, 1]], dtype=np.float32),
            "max_height": 1.0,
            "ds_per_pixel": 0.25,
            "Mm_per_ds": 1.0,
            "Gauss_per_dB": 100.0,
            "wcs": [],
        },
    }, path)


def _constant_spherical_checkpoint(path):
    model = BModel(dim=8, n_layers=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.out_layer.bias[2] = 1
    torch.save({
        "model": model,
        "data": {
            "type": "spherical",
            "radius_range": np.array([1.0, 2.0]),
            "Mm_per_ds": 1.0,
            "Gauss_per_dB": 100.0,
        },
    }, path)


def test_public_loader_loads_checkpoint_once_and_tracer_reuses_model(tmp_path):
    checkpoint = tmp_path / "constant.nf2"
    _constant_checkpoint(checkpoint)

    original_load = torch.load
    with patch("torch.load", wraps=original_load) as mocked_load:
        output = nf2.load(checkpoint, device="cpu")
    assert mocked_load.call_count == 1

    metrics = output.compute_fieldline_metrics(
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        metrics=[
            "squashing_factor", "twist_number", "fieldline_length",
            "integrated_current_density", "fieldline_geometry",
        ],
        trace_config={"method": "rk4", "step_size_Mm": 0.1, "max_steps": 20},
    )

    np.testing.assert_allclose(metrics["squashing_factor"], 2, atol=2e-5)
    np.testing.assert_allclose(metrics["twist_number"].value, 0, atol=1e-7)
    np.testing.assert_allclose(metrics["fieldline_length"].to_value(u.Mm), 1, atol=2e-5)
    np.testing.assert_allclose(metrics["integrated_current_density"].value, 0, atol=1e-7)
    np.testing.assert_allclose(metrics["apex_height"].to_value(u.Mm), 1, atol=2e-5)
    assert metrics["open"].item()
    assert metrics["open_polarity"].item() == 1


def test_npz_export_includes_fieldline_metrics(tmp_path):
    checkpoint = tmp_path / "constant.nf2"
    export_path = tmp_path / "constant.npz"
    _constant_checkpoint(checkpoint)

    nf2.export_file(
        checkpoint,
        export_path,
        fmt="npz",
        Mm_per_pixel=0.5,
        height_range=[0, 1],
        metrics=["squashing_factor", "twist_number", "fieldline_length", "fieldline_geometry"],
        trace_config={"method": "rk4", "step_size_Mm": 0.1, "max_steps": 20},
        progress=False,
    )

    with np.load(export_path) as exported:
        assert {
            "squashing_factor", "log10_q", "q_valid", "q_condition_number",
            "twist_number", "fieldline_length", "open", "closed", "open_polarity",
            "footpoint_separation", "apex_height",
        }.issubset(exported.files)
        np.testing.assert_allclose(exported["squashing_factor"], 2, atol=2e-5)


def test_derivative_metric_enables_jacobian_when_sampling_disables_it(tmp_path):
    checkpoint = tmp_path / "constant.nf2"
    _constant_checkpoint(checkpoint)
    output = nf2.load(checkpoint, device="cpu")

    result = output.load_coords(
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        compute_jacobian=False,
        metrics=["j"],
    )

    assert "jac_matrix" in result
    np.testing.assert_allclose(result["metrics"]["j"].value, 0, atol=1e-7)


def test_model_evaluation_passes_parallel_flags_positionally(tmp_path):
    checkpoint = tmp_path / "constant.nf2"
    _constant_checkpoint(checkpoint)
    output = nf2.load(checkpoint, device="cpu")
    model = output.model

    class ParallelCallRecorder:
        def __call__(self, *args, **kwargs):
            assert len(args) == 2
            assert kwargs == {}
            return model(*args)

    output.model = ParallelCallRecorder()
    result = output._sample_tensor([[0.5, 0.5, 0.5]], compute_jacobian=False)

    assert result["b"].shape == (1, 3)


def test_spherical_loader_returns_spherical_field_components(tmp_path):
    checkpoint = tmp_path / "constant_spherical.nf2"
    _constant_spherical_checkpoint(checkpoint)
    output = nf2.load(checkpoint, device="cpu")

    result = output.load_spherical(
        radius_range=np.array([1.0, 1.0]) * u.solRad,
        latitude_range=np.array([90.0, 90.0]) * u.deg,
        longitude_range=np.array([0.0, 0.0]) * u.deg,
        sampling=(1, 1, 1),
        compute_jacobian=False,
    )

    assert result["b_rtp"].shape == result["b"].shape == (1, 1, 1, 3)
    np.testing.assert_allclose(result["b_rtp"].to_value(u.G), [[[[100, 0, 0]]]], atol=1e-5)


def test_spherical_layer_supports_uniform_sine_latitude_sampling(tmp_path):
    checkpoint = tmp_path / "constant_spherical.nf2"
    _constant_spherical_checkpoint(checkpoint)
    output = nf2.load(checkpoint, device="cpu")

    result = output.load_spherical_layer(
        radius=1 * u.solRad,
        latitude_range=(-90, 90) * u.deg,
        longitude_range=(0, 0) * u.deg,
        sampling=(5, 1),
        sin_latitude=True,
        compute_jacobian=False,
    )

    latitude = np.pi / 2 - result["spherical_coords"][..., 1]
    sin_latitude_edges = np.linspace(1, -1, 6)
    expected_centers = (sin_latitude_edges[:-1] + sin_latitude_edges[1:]) / 2
    np.testing.assert_allclose(np.sin(latitude[:, 0]), expected_centers, atol=1e-7)
