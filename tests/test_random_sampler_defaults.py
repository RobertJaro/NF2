from nf2.train.config import DEFAULT_DATA_ITERATIONS, normalize_config
from nf2.loader.spherical import SphericalDataModule


def test_cartesian_data_iterations_default_is_release_scale():
    config = normalize_config(
        {
            "data": {
                "geometry": "cartesian",
                "boundaries": [{"type": "analytical", "case": 1}],
            },
        }
    )

    assert config["data"]["iterations"] == DEFAULT_DATA_ITERATIONS


def test_spherical_random_sampler_length_uses_default_data_iterations():
    config = normalize_config(
        {
            "data": {
                "geometry": "spherical",
                "boundaries": [{"id": "full_disk", "type": "map", "files": {"Br": "br.fits"}}],
            },
            "losses": [
                {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": ["full_disk"]},
            ],
        }
    )

    assert config["data"]["iterations"] == DEFAULT_DATA_ITERATIONS
    assert config["data"]["samplers"][0]["length"] == DEFAULT_DATA_ITERATIONS


def test_data_iterations_override_is_preserved():
    config = normalize_config(
        {
            "data": {
                "geometry": "cartesian",
                "boundaries": [{"type": "analytical", "case": 1}],
                "iterations": 7,
            },
        }
    )

    assert config["data"]["iterations"] == 7


def test_spherical_data_module_accepts_top_level_iterations(tmp_path):
    data_module = SphericalDataModule(
        boundaries=[{"id": "random", "type": "random_spherical"}],
        validation=[{"id": "sphere", "type": "sphere"}],
        iterations=7,
        work_path=tmp_path,
    )

    assert len(data_module.training_datasets["random"]) == 7
