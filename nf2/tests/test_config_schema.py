from nf2.config.schema import canonical_to_runtime_config


def test_canonical_cartesian_single_config_maps_to_runtime():
    config = {
        "run": {
            "mode": "single",
            "geometry": "cartesian",
            "output_dir": "/tmp/out",
            "work_dir": "/tmp/work",
        },
        "logging": {"project": "demo"},
        "data": {
            "parameters": {"iterations": 10},
            "train": [{"type": "fits", "fits_path": {"Br": "a", "Bt": "b", "Bp": "c"}}],
            "validation": [{"type": "cube", "ds_id": "cube"}],
        },
        "model": {"type": "b", "dim": 32},
        "training": {"epochs": 1},
        "losses": [{"type": "force_free", "lambda": 0.1}],
    }

    runtime = canonical_to_runtime_config(config)

    assert runtime["base_path"] == "/tmp/out"
    assert runtime["work_directory"] == "/tmp/work"
    assert runtime["data"]["type"] == "cartesian"
    assert runtime["data"]["train_configs"][0]["type"] == "fits"
    assert runtime["data"]["valid_configs"][0]["type"] == "cube"


def test_canonical_spherical_series_config_maps_to_runtime():
    config = {
        "run": {
            "mode": "series",
            "geometry": "spherical",
            "output_dir": "/tmp/out",
            "work_dir": "/tmp/work",
            "resume_from": "/tmp/init.ckpt",
        },
        "data": {
            "parameters": {"iterations": 10, "max_radius": 1.3},
            "train": [],
            "validation": [{"type": "sphere", "ds_id": "sphere"}],
            "sequence": {
                "frames": [{"Br": "a", "Bt": "b", "Bp": "c"}],
                "synoptic": {"Br": "d", "Bt": "e", "Bp": "f"},
            },
        },
        "model": {"type": "vector_potential", "dim": 64},
        "training": {},
        "losses": [],
    }

    runtime = canonical_to_runtime_config(config)

    assert runtime["meta_path"] == "/tmp/init.ckpt"
    assert runtime["data"]["type"] == "spherical"
    assert runtime["data"]["fits_paths"][0]["Br"] == "a"
    assert runtime["data"]["synoptic_fits_path"]["Br"] == "d"
