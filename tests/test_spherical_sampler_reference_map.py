import numpy as np
import pytest

from nf2.loader import spherical
from nf2.loader import spherical_datasets


@pytest.mark.parametrize("sampler_type", ["random_spherical", "random_radial_grouped"])
def test_random_sampler_uses_reference_map_bounds(monkeypatch, tmp_path, sampler_type):
    reference_map = object()
    captured = {}

    monkeypatch.setattr(spherical, "_map", lambda path: reference_map)

    def coordinate_bounds(smap, mu_filter, name):
        captured.update(smap=smap, mu_filter=mu_filter, name=name)
        return {
            "latitude_range": [-61.5, 72.25],
            "longitude_range": [301.5, 463.5],
        }

    monkeypatch.setattr(spherical, "_reference_map_coordinate_bounds", coordinate_bounds)

    data_module = spherical.SphericalDataModule(
        boundaries=[],
        samplers=[{
            "id": "random",
            "type": sampler_type,
            "batch_size": 64,
            "n_lat_lon_sample": 8,
            "reference_map": {
                "file": "full_disk.fits",
                "mu_filter": {"min": 0.2},
            },
        }],
        validation=[],
        work_path=tmp_path,
    )

    dataset = data_module.training_datasets["random"]
    assert dataset.colatitude_range == pytest.approx(np.deg2rad([17.75, 151.5]))
    assert dataset.longitude_range == pytest.approx(np.deg2rad([301.5, 463.5]))
    assert captured == {
        "smap": reference_map,
        "mu_filter": {"min": 0.2},
        "name": "reference_map file: full_disk.fits",
    }
    assert "reference_map" not in dataset.config


def test_explicit_sampler_bounds_override_reference_map(monkeypatch):
    monkeypatch.setattr(spherical, "_map", lambda path: object())
    monkeypatch.setattr(
        spherical,
        "_reference_map_coordinate_bounds",
        lambda *args: {"latitude_range": [-60, 60], "longitude_range": [300, 460]},
    )
    config = {
        "reference_map": "full_disk.fits",
        "latitude_range": [-10, 20],
        "longitude_range": [30, 40],
    }

    spherical.SphericalDataModule._apply_random_reference_map(config)

    assert config == {
        "latitude_range": [-10, 20],
        "longitude_range": [30, 40],
        "unit": "deg",
    }


def test_reference_map_bounds_use_finite_unmasked_pixels(monkeypatch):
    reference_map = type("ReferenceMap", (), {
        "data": np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    })()
    longitude = np.array([[350.0, 355.0, np.nan], [365.0, 370.0, 375.0]])
    latitude = np.array([[-30.0, -20.0, -10.0], [10.0, 20.0, 30.0]])
    mu_mask = np.array([[False, False, False], [False, True, False]])

    monkeypatch.setattr(
        spherical_datasets,
        "_reference_map_lon_lat",
        lambda smap: (longitude, latitude),
    )
    monkeypatch.setattr(
        spherical_datasets,
        "_mu_filter_mask",
        lambda smap, mu_filter: mu_mask,
    )

    bounds = spherical_datasets._reference_map_coordinate_bounds(
        reference_map,
        {"min": 0.2},
    )

    assert bounds == {
        "latitude_range": [-30.0, 30.0],
        "longitude_range": [350.0, 375.0],
    }


def test_reference_map_bounds_reject_empty_selection(monkeypatch):
    reference_map = type("ReferenceMap", (), {"data": np.array([[np.nan]])})()
    monkeypatch.setattr(
        spherical_datasets,
        "_reference_map_lon_lat",
        lambda smap: (np.array([[0.0]]), np.array([[0.0]])),
    )

    with pytest.raises(ValueError, match="No valid coordinates"):
        spherical_datasets._reference_map_coordinate_bounds(reference_map)


@pytest.mark.parametrize("reference_map", [{}, [], 42])
def test_invalid_reference_map_config_is_rejected(reference_map):
    config = {"reference_map": reference_map}

    with pytest.raises((TypeError, ValueError)):
        spherical.SphericalDataModule._apply_random_reference_map(config)
