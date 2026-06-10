import numpy as np
from astropy import units as u

from nf2.evaluation.output_metrics import OUTPUT_METRICS, metric_mapping, normalize_metric_names
from nf2.evaluation.vtk import split_vectors_scalars
from nf2.reference import EXPORT_METRICS


def test_output_metric_registry_drives_mapping_and_reference():
    assert set(metric_mapping) == set(OUTPUT_METRICS)

    reference_rows = {
        name: (outputs, description)
        for name, outputs, description in EXPORT_METRICS
    }
    assert set(reference_rows) == set(OUTPUT_METRICS)

    for name, metric in OUTPUT_METRICS.items():
        assert metric_mapping[name] is metric.func
        assert reference_rows[name] == (", ".join(metric.outputs), metric.description)


def test_output_metric_keys_are_unambiguous():
    assert OUTPUT_METRICS["j"].outputs == ("j",)
    assert "j_vec" not in OUTPUT_METRICS
    assert OUTPUT_METRICS["energy_gradient"].outputs == ("energy_gradient",)
    assert OUTPUT_METRICS["spherical_energy_gradient"].outputs == ("spherical_energy_gradient",)
    assert OUTPUT_METRICS["free_energy"].outputs == ("free_energy",)
    assert OUTPUT_METRICS["free_energy_fft"].outputs == ("free_energy_fft",)
    assert OUTPUT_METRICS["free_energy_direct"].outputs == ("free_energy_direct",)
    assert OUTPUT_METRICS["los_trv_azi"].outputs == ("los_trv_azi",)


def test_output_metric_functions_return_declared_keys_for_core_metrics():
    b = np.ones((2, 2, 2, 3)) * u.G
    jac_matrix = np.zeros((2, 2, 2, 3, 3)) * u.G / u.Mm
    coords = np.ones((2, 2, 2, 3))
    a = np.ones((2, 2, 2, 3)) * u.G * u.m
    state = {"b": b, "jac_matrix": jac_matrix, "coords": coords, "a": a}

    optional_or_heavy = {"free_energy_direct", "squashing_factor"}
    for name, metric in OUTPUT_METRICS.items():
        if name in optional_or_heavy:
            continue
        assert set(metric.func(**state)) == set(metric.outputs)

    assert metric_mapping["j"](**state)["j"].shape == (2, 2, 2, 3)


def test_normalize_metric_names_accepts_none_string_and_iterables():
    assert normalize_metric_names(None) == []
    assert normalize_metric_names("alpha") == ["alpha"]
    assert normalize_metric_names(("j", "energy")) == ["j", "energy"]


def test_vtk_metric_split_treats_j_as_vector():
    vectors, scalars = split_vectors_scalars({
        "j": np.zeros((2, 2, 2, 3)),
    })

    assert scalars == {}
    assert set(vectors) == {"j"}
