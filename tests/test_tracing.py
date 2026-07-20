from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nf2.evaluation.tracing import (
    BatchedFieldLineTracer,
    CartesianTraceGeometry,
    SphericalTraceGeometry,
    TraceConfig,
)


def _output(sample):
    return SimpleNamespace(device=torch.device("cpu"), _sample_tensor=sample)


def _uniform_sample(coords, compute_jacobian=False):
    b = torch.zeros_like(coords)
    b[:, 2] = 1
    result = {"b": b}
    if compute_jacobian:
        result["jac_matrix"] = torch.zeros((*coords.shape[:-1], 3, 3), dtype=coords.dtype)
    return result


def test_default_integrator_is_rkf45():
    assert TraceConfig().method == "rkf45"


@pytest.mark.parametrize("method", ["euler", "rk1", "rk2", "rk3", "rk4", "rkf45"])
def test_uniform_cartesian_field_all_integrators(method):
    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(method=method, step_size=0.1, max_steps=20),
    )

    result = tracer.trace(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), metrics=["fieldline_length"])

    np.testing.assert_allclose(result["fieldline_length"], 1, atol=2e-6)
    np.testing.assert_allclose(result["backward_endpoint"], [[0.5, 0.5, 0]], atol=2e-6)
    np.testing.assert_allclose(result["forward_endpoint"], [[0.5, 0.5, 1]], atol=2e-6)
    assert result["backward_boundary"].item() == 4
    assert result["forward_boundary"].item() == 5


@pytest.mark.parametrize("q_method", ["tangent", "perturbed"])
def test_uniform_field_has_minimum_squashing(q_method):
    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, q_method=q_method, q_epsilon=1e-3),
    )

    result = tracer.trace([[0.5, 0.5, 0.5]], metrics=["squashing_factor"])

    np.testing.assert_allclose(result["squashing_factor"], 2, atol=2e-5)
    assert result["q_valid"].item()


def test_perturbed_q_uses_boundary_tangent_stencil_for_boundary_seed():
    def sample(coords, compute_jacobian=False):
        b = torch.zeros_like(coords)
        b[:, 0] = 0.25
        b[:, 2] = 1
        result = {"b": b}
        if compute_jacobian:
            result["jac_matrix"] = torch.zeros((coords.shape[0], 3, 3), dtype=coords.dtype)
        return result

    tracer = BatchedFieldLineTracer(
        _output(sample),
        CartesianTraceGeometry([[-2, 2], [-2, 2], [0, 1]]),
        TraceConfig(step_size=0.05, max_steps=100, q_method="perturbed", q_epsilon=1e-3),
    )
    result = tracer.trace([0, 0, 0], metrics=["squashing_factor"])

    np.testing.assert_allclose(result["squashing_factor"], 2, atol=2e-4)
    assert result["q_valid"].item()


def test_perturbed_q_traces_only_four_displaced_lines_per_seed():
    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, q_method="perturbed", q_epsilon=1e-3),
    )

    with patch.object(tracer, "_trace_bidirectional_queued", wraps=tracer._trace_bidirectional_queued) as trace:
        tracer.trace([[0.5, 0.5, 0.5]], metrics=["squashing_factor"])

    assert [call.args[0].shape[0] for call in trace.call_args_list] == [1, 4]


def test_twist_and_integrated_current_for_constant_curl_axis():
    a = 0.4

    def sample(coords, compute_jacobian=False):
        b = torch.stack((-a * coords[:, 1], a * coords[:, 0], torch.ones_like(coords[:, 0])), -1)
        result = {"b": b}
        if compute_jacobian:
            jac = torch.zeros((coords.shape[0], 3, 3), dtype=coords.dtype)
            jac[:, 0, 1] = -a
            jac[:, 1, 0] = a
            result["jac_matrix"] = jac
        return result

    tracer = BatchedFieldLineTracer(
        _output(sample),
        CartesianTraceGeometry([[-1, 1], [-1, 1], [0, 1]]),
        TraceConfig(step_size=0.02, max_steps=100),
    )
    result = tracer.trace(
        [[0, 0, 0.5]], metrics=["twist_number", "integrated_current_density", "fieldline_length"]
    )

    np.testing.assert_allclose(result["fieldline_length"], 1, atol=2e-5)
    np.testing.assert_allclose(result["twist_number"], a / (2 * np.pi), rtol=1e-5)
    np.testing.assert_allclose(result["integrated_current_density"], [[0, 0, 2 * a]], atol=2e-5)


def test_radial_spherical_connectivity_and_geometry():
    def sample(coords, compute_jacobian=False):
        radius = torch.linalg.vector_norm(coords, dim=-1, keepdim=True)
        result = {"b": coords / radius}
        if compute_jacobian:
            raise AssertionError("Jacobian should not be requested for geometry-only tracing")
        return result

    tracer = BatchedFieldLineTracer(
        _output(sample),
        SphericalTraceGeometry((1, 2)),
        TraceConfig(step_size=0.05, max_steps=100),
    )
    result = tracer.trace([[1.5, 0, 0]], metrics=["fieldline_length", "fieldline_geometry"])

    np.testing.assert_allclose(result["fieldline_length"], 1, atol=2e-5)
    np.testing.assert_allclose(result["footpoint_separation"], 1, atol=2e-5)
    np.testing.assert_allclose(result["apex"], 2, atol=2e-5)
    assert result["open"].item()
    assert not result["closed"].item()
    assert result["open_polarity"].item() == 1


@pytest.mark.parametrize("q_method", ["tangent", "perturbed"])
def test_radial_spherical_field_has_minimum_squashing(q_method):
    def sample(coords, compute_jacobian=False):
        radius = torch.linalg.vector_norm(coords, dim=-1, keepdim=True)
        radial = coords / radius
        result = {"b": radial}
        if compute_jacobian:
            identity = torch.eye(3, dtype=coords.dtype).expand(coords.shape[0], -1, -1)
            result["jac_matrix"] = (identity - radial[:, :, None] * radial[:, None, :]) / radius[:, None]
        return result

    tracer = BatchedFieldLineTracer(
        _output(sample),
        SphericalTraceGeometry((1, 2)),
        TraceConfig(step_size=0.025, max_steps=100, q_method=q_method, q_epsilon=1e-3),
    )
    result = tracer.trace([[1.5, 0, 0]], metrics=["squashing_factor"])

    np.testing.assert_allclose(result["squashing_factor"], 2, atol=2e-3)
    assert result["q_valid"].item()


def test_persistent_slot_queue_is_independent_of_slot_capacity():
    seeds = np.stack([
        np.full(9, 0.5),
        np.full(9, 0.5),
        np.array([0.05, 0.2, 0.4, 0.9, 0.1, 0.7, 0.3, 0.8, 0.6]),
    ], axis=-1).astype(np.float32)
    geometry = CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]])
    metrics = ["squashing_factor", "twist_number", "integrated_current_density", "fieldline_geometry"]

    queued = BatchedFieldLineTracer(
        _output(_uniform_sample), geometry,
        TraceConfig(step_size=0.05, max_steps=100, batch_size=2),
    ).trace(seeds, metrics=metrics)
    unbounded = BatchedFieldLineTracer(
        _output(_uniform_sample), geometry,
        TraceConfig(step_size=0.05, max_steps=100, batch_size=100),
    ).trace(seeds, metrics=metrics)

    for key in (
        "forward_endpoint", "backward_endpoint", "fieldline_length", "twist_number",
        "integrated_current_density", "squashing_factor", "open", "closed", "open_polarity",
    ):
        np.testing.assert_allclose(queued[key], unbounded[key], atol=2e-5)


def test_tracing_progress_counts_completed_field_lines():
    bars = []

    class ProgressRecorder:
        def __init__(self, total, **kwargs):
            self.total = total
            self.unit = kwargs["unit"]
            self.completed = 0
            self.closed = False
            bars.append(self)

        def update(self, count):
            self.completed += count

        def close(self):
            self.closed = True

    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, batch_size=1, progress=True),
    )
    with patch("nf2.evaluation.tracing.tracer.tqdm", ProgressRecorder):
        tracer.trace([[0.5, 0.5, 0.5]], metrics=["fieldline_length"])

    assert len(bars) == 1
    assert all(bar.completed == bar.total == 1 for bar in bars)
    assert bars[0].unit == "lines"
    assert all(bar.closed for bar in bars)


def test_stored_path_combines_backward_and_forward_halves():
    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, batch_size=1, store_path=True),
    )
    result = tracer.trace([[0.5, 0.5, 0.5]], metrics=["fieldline_length"])
    path = result["path"][:, 0]
    path = path[np.isfinite(path).all(-1)]

    np.testing.assert_allclose(path[0], [0.5, 0.5, 0], atol=2e-6)
    np.testing.assert_allclose(path[-1], [0.5, 0.5, 1], atol=2e-6)
    assert np.all(np.diff(path[:, 2]) > 0)
    assert np.count_nonzero(np.isclose(path[:, 2], 0.5)) == 1


def test_max_length_and_boundary_termination_are_exact():
    tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, max_length=0.23),
    )

    result = tracer.trace([[0.5, 0.5, 0.5]], metrics=["fieldline_length"])

    np.testing.assert_allclose(result["fieldline_length"], 0.46, atol=2e-7)
    np.testing.assert_allclose(result["backward_endpoint"], [[0.5, 0.5, 0.27]], atol=2e-7)
    np.testing.assert_allclose(result["forward_endpoint"], [[0.5, 0.5, 0.73]], atol=2e-7)
    assert result["backward_status"].item() == result["status_codes"]["max_length"]
    assert result["forward_status"].item() == result["status_codes"]["max_length"]

    boundary_tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20, max_length=0.08),
    )
    boundary_result = boundary_tracer.trace([[0.5, 0.5, 0.95]], metrics=["fieldline_length"])

    np.testing.assert_allclose(boundary_result["fieldline_length"], 0.13, atol=2e-7)
    np.testing.assert_allclose(boundary_result["backward_endpoint"], [[0.5, 0.5, 0.87]], atol=2e-7)
    np.testing.assert_allclose(boundary_result["forward_endpoint"], [[0.5, 0.5, 1]], atol=2e-7)
    assert boundary_result["backward_status"].item() == boundary_result["status_codes"]["max_length"]
    assert boundary_result["forward_status"].item() == boundary_result["status_codes"]["boundary"]


def test_trace_input_validation_and_empty_seeds():
    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("An empty trace must not sample the model")

    tracer = BatchedFieldLineTracer(
        _output(fail_if_sampled),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(store_path=True),
    )

    result = tracer.trace(
        np.empty((0, 3), dtype=np.float32),
        metrics=["squashing_factor", "twist_number", "integrated_current_density"],
    )

    assert result["fieldline_length"].shape == (0,)
    assert result["integrated_current_density"].shape == (0, 3)
    assert result["squashing_factor"].shape == (0,)
    assert result["path"].shape == (0, 0, 3)

    validation_tracer = BatchedFieldLineTracer(
        _output(_uniform_sample),
        CartesianTraceGeometry([[0, 1], [0, 1], [0, 1]]),
        TraceConfig(step_size=0.1, max_steps=20),
    )

    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        validation_tracer.trace(np.zeros((2, 2), dtype=np.float32))

    result = validation_tracer.trace([[0.5, 0.5, 0.5]], metrics="twist_number")

    np.testing.assert_allclose(result["twist_number"], 0, atol=1e-7)
    with pytest.raises(ValueError, match="Unknown field-line metric"):
        validation_tracer.trace([[0.5, 0.5, 0.5]], metrics=["typo"])
