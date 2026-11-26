import csv
import json

import numpy as np
import pytest
import torch

from nf2.cli import main as cli
from nf2.evaluation import energy, helicity
from nf2.evaluation.output import CartesianOutput
from nf2.metrics import compute_metrics
from nf2.train.model import BModel


@pytest.fixture
def checkpoint(tmp_path):
    model = BModel(dim=2, n_layers=1)
    with torch.no_grad():
        model.out_layer.weight.zero_()
        model.out_layer.bias.copy_(torch.tensor([0., 0., 1.]))
    state = {"model": model, "data": {"type": "cartesian", "Gauss_per_dB": 2., "Mm_per_ds": 1.,
             "ds_per_pixel": [0.25], "coord_range": [np.array([[0., 1.], [0., 1.]])], "max_height": 1.}}
    path = tmp_path / "box.nf2"
    torch.save(state, path)
    return path


def test_default_table_and_energy_reference(checkpoint, capsys):
    cli(["metrics", str(checkpoint)])
    report = capsys.readouterr().out
    assert "E_free" in report and "rms_divB" in report
    assert "mean_B" not in report and "helicity" not in report and "net_flux" not in report
    result = compute_metrics(checkpoint)
    assert result["E_tot"] == pytest.approx(4e24 / (8 * np.pi))
    assert result["E_pot"] == pytest.approx(result["E_tot"])
    assert abs(result["E_free"]) < 1e-6 * result["E_tot"]
    assert result["theta_J"] == result["rms_divB"] == 0


def test_flux_on_cropped_faces_only(checkpoint, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Flux requested a volume or potential solve")
    monkeypatch.setattr(CartesianOutput, "load_cube", unexpected)
    monkeypatch.setattr(helicity, "_potential_reference", unexpected)
    result = compute_metrics(checkpoint, metrics=["flux"], x_range=[0, .5], y_range=[.2, 1.])
    assert result["bottom_unsigned_flux_Mx"] == pytest.approx(.8e16)
    assert result["unsigned_flux_Mx"] == pytest.approx(1.6e16)
    assert result["net_flux_Mx"] == result["flux_imbalance_fraction"] == 0


def test_selection_skips_fft_and_jacobian(checkpoint, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Unexpected FFT or field Jacobian")
    monkeypatch.setattr(energy, "get_free_mag_energy", unexpected)
    monkeypatch.setattr("nf2.train.model.jacobian", unexpected)
    result = compute_metrics(checkpoint, metrics=["mean_B", "E_tot", "mean_B"])
    assert list(result) == ["geometry", "mean_B", "E_tot"]
    assert result["mean_B"] == pytest.approx(2.)


def test_combined_cli_report_formats(checkpoint, tmp_path, capsys):
    request = ["metrics", str(checkpoint), "--metrics", "CWsin", "quality", "flux", "helicity",
               "--height_range", ".2", ".8", "--Mm_per_pixel", ".1"]
    for ext in ("txt", "csv", "json"):
        cli([*request, "--output", str(tmp_path / f"report.{ext}")])
    assert not capsys.readouterr().out
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["grids"]["closed_box"]["box_Mm"][2] == [.2, .8]
    with (tmp_path / "report.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [row["metric"] for row in rows] == [m["metric"] for m in data["metrics"]]
    assert sum(row["metric"] == "sigma_J" for row in rows) == 1
    for row, metric in zip(rows, data["metrics"]):
        assert float(row["value"]) == metric["value"]
        assert json.loads(row["sampling"]) == data["grids"][row["grid"]]
        assert row["metric"] in (tmp_path / "report.txt").read_text()
        if row["metric"].startswith("H_"):
            assert row["direction"] in ("top", "bottom") and row["unit"] == "Mx²"
            assert abs(metric["value"]) < 1e20


def test_spherical_selection_and_errors(checkpoint, capsys):
    state = torch.load(checkpoint, weights_only=False)
    state["data"].update(type="spherical", radius_range=[1., 1.2])
    torch.save(state, checkpoint)
    result = compute_metrics(checkpoint, spherical_sampling=[3, 5, 7])
    assert result["geometry"] == "spherical" and result["E_tot"] > 0 and "E_pot" not in result
    for name in ["flux", "H_V_bottom", "E_free"]:
        with pytest.raises(ValueError, match="Cartesian"):
            compute_metrics(checkpoint, metrics=[name])
    cli(["metrics", "--list-metrics"])
    assert "flux_imbalance_fraction" in capsys.readouterr().out
    for args, msg in [(["--metrics", "bogus"], "Unknown metric"), (["--output", "bad.tsv"], "--output")]:
        with pytest.raises(SystemExit) as error:
            cli(["metrics", str(checkpoint), *args])
        assert error.value.code == 2 and msg in capsys.readouterr().err


def test_spherical_free_boundary_evaluates_exterior_field_and_jacobian():
    from nf2.evaluation.field import evaluate_field
    from nf2.train.model import FreeBoundaryVectorPotentialModel

    model = FreeBoundaryVectorPotentialModel(Mm_per_ds=695.7, dim=4, n_layers=1)
    coords = torch.tensor([[1.2, 0., 0.], [2.5, 0., 0.], [3., 0., 0.]], requires_grad=True)
    result = evaluate_field(model, coords)
    assert result["inside_source_surface"].flatten().tolist() == [True, False, False]
    surface = torch.tensor([[2., 0., 0.]], requires_grad=True)
    expected_br = model(surface, compute_jacobian=False)["b"][0, 0] * (2 / coords[1:, 0]) ** 2
    torch.testing.assert_close(result["b"][1:, 0], expected_br)
    torch.testing.assert_close(result["b"][1:, 1:], torch.zeros(2, 2))
    torch.testing.assert_close(result["jac_matrix"][1:, :, 0], -2 * result["b"][1:] / coords[1:, :1])
