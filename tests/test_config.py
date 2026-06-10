import pytest

from nf2.train.util import _resolve_bundled_config, load_yaml_config
from nf2.train.config import normalize_config


def _spherical_config(boundary):
    return {
        "data": {
            "geometry": "spherical",
            "boundaries": [boundary],
        },
        "losses": [
            {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": ["full_disk"]},
        ],
    }


def test_map_errors_merge_into_file_mapping():
    config = normalize_config(
        _spherical_config(
            {
                "id": "full_disk",
                "type": "map",
                "files": {"Br": "br.fits", "Bt": "bt.fits", "Bp": "bp.fits"},
                "errors": {"Br_err": "br_err.fits", "Bt_err": "bt_err.fits", "Bp_err": "bp_err.fits"},
            }
        )
    )

    files = config["data"]["boundaries"][0]["files"]
    assert files["Br"] == "br.fits"
    assert files["Br_err"] == "br_err.fits"


def test_load_bundled_config_template_by_name():
    config = load_yaml_config(
        "nf2/cartesian/minimal_fits.yaml",
        ["--run_path", "./runs/minimal", "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits"],
    )

    assert config["path"] == "./runs/minimal"
    assert config["data"]["boundaries"][0]["fits_path"]["Br"] == "Br.fits"


def test_default_cartesian_config_uses_100_mm_z_range():
    with pytest.warns(UserWarning, match="Skipping optional error-file configuration"):
        config = load_yaml_config(
            "nf2/cartesian/sharp_cea.yaml",
            ["--run_path", "./runs/sharp", "--work_path", "./runs/sharp/work",
             "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits"],
        )

    assert config["data"]["z_range"] == [0, 100]


def test_cartesian_z_range_can_be_overridden_from_cli_args():
    with pytest.warns(UserWarning, match="Skipping optional error-file configuration"):
        config = load_yaml_config(
            "nf2/cartesian/sharp_cea.yaml",
            ["--run_path", "./runs/sharp", "--work_path", "./runs/sharp/work",
             "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits",
             "--z_range", "0", "150"],
        )

    assert config["data"]["z_range"] == [0.0, 150.0]


def test_default_cartesian_potential_start_matches_force_free_weight():
    config = load_yaml_config(
        "nf2/cartesian/minimal_fits.yaml",
        ["--run_path", "./runs/minimal", "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits"],
    )

    losses = {loss["name"]: loss for loss in config["losses"]}
    assert losses["potential"]["weight"]["start"] == losses["force_free"]["weight"]


def test_cartesian_force_free_weight_can_be_overridden_from_cli_args():
    config = load_yaml_config(
        "nf2/cartesian/minimal_fits.yaml",
        ["--run_path", "./runs/minimal", "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits",
         "--force_free_weight", "2.0e-3"],
    )

    losses = {loss["name"]: loss for loss in config["losses"]}
    assert losses["force_free"]["weight"] == 2.0e-3
    assert losses["potential"]["weight"]["start"] == 2.0e-3


def test_missing_required_placeholder_raises_error():
    with pytest.raises(ValueError, match="Bp"):
        load_yaml_config(
            "nf2/cartesian/minimal_fits.yaml",
            ["--run_path", "./runs/minimal", "--Br", "Br.fits", "--Bt", "Bt.fits"],
        )


def test_load_bundled_config_template_from_documented_examples_path():
    template = _resolve_bundled_config("examples/configs/cartesian/minimal_fits.yaml")

    assert template is not None
    assert template.name == "minimal_fits.yaml"


def test_scaled_vector_potential_model_field_is_supported():
    config = normalize_config(
        {
            "data": {
                "geometry": "spherical",
                "boundaries": [{"id": "full_disk", "type": "map", "files": {"Br": "br.fits"}}],
            },
            "model": {"field": "scaled_vector_potential"},
            "losses": [
                {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": ["full_disk"]},
            ],
        }
    )

    assert config["model"]["type"] == "scaled_vector_potential"


def test_unset_bundled_error_placeholders_are_skipped():
    with pytest.warns(UserWarning, match="Skipping optional error-file configuration"):
        config = load_yaml_config(
            "nf2/cartesian/sharp_cea.yaml",
            ["--run_path", "./runs/sharp", "--work_path", "./runs/sharp/work",
             "--Br", "Br.fits", "--Bt", "Bt.fits", "--Bp", "Bp.fits"],
        )

    boundary = config["data"]["boundaries"][0]
    validation = config["data"]["validation"][0]
    assert "error_path" not in boundary
    assert "error_path" not in validation


def test_set_bundled_error_placeholders_are_kept():
    config = load_yaml_config(
        "nf2/cartesian/sharp_cea.yaml",
        [
            "--run_path", "./runs/sharp",
            "--work_path", "./runs/sharp/work",
            "--Br", "Br.fits",
            "--Bt", "Bt.fits",
            "--Bp", "Bp.fits",
            "--Br_err", "Br_err.fits",
            "--Bt_err", "Bt_err.fits",
            "--Bp_err", "Bp_err.fits",
        ],
    )

    assert config["data"]["boundaries"][0]["error_path"]["Br_err"] == "Br_err.fits"
    assert config["data"]["validation"][0]["error_path"]["Bp_err"] == "Bp_err.fits"


def test_dependent_error_references_are_skipped_when_source_errors_are_skipped():
    with pytest.warns(UserWarning, match="Skipping optional error-file"):
        config = load_yaml_config(
            "nf2/spherical/hmi_full_disk_series.yaml",
            [
                "--run_path", "./runs/spherical_series",
                "--work_path", "./runs/spherical_series/work",
                "--meta_path", "./runs/spherical_initial/last.ckpt",
                "--wandb_project", "nf2",
                "--run_name", "spherical series",
                "--full_disk_Br_pattern", "full_disk/*.Br.fits",
                "--full_disk_Bt_pattern", "full_disk/*.Bt.fits",
                "--full_disk_Bp_pattern", "full_disk/*.Bp.fits",
                "--synoptic_Br_pattern", "synoptic/*.Br.fits",
                "--synoptic_Bt_pattern", "synoptic/*.Bt.fits",
                "--synoptic_Bp_pattern", "synoptic/*.Bp.fits",
            ],
        )

    assert "Br_err" not in config["data"]["boundaries"][0]["files"]
    assert "Br_err" not in config["data"]["validation"][0]["files"]


def test_map_errors_merge_into_file_series():
    config = normalize_config(
        _spherical_config(
            {
                "id": "full_disk",
                "type": "map",
                "files": [
                    {"Br": "br_1.fits", "Bt": "bt_1.fits", "Bp": "bp_1.fits"},
                    {"Br": "br_2.fits", "Bt": "bt_2.fits", "Bp": "bp_2.fits"},
                ],
                "errors": [
                    {"Br_err": "br_err_1.fits", "Bt_err": "bt_err_1.fits", "Bp_err": "bp_err_1.fits"},
                    {"Br_err": "br_err_2.fits", "Bt_err": "bt_err_2.fits", "Bp_err": "bp_err_2.fits"},
                ],
            }
        )
    )

    files = config["data"]["boundaries"][0]["files"]
    assert files[0]["Br"] == "br_1.fits"
    assert files[0]["Br_err"] == "br_err_1.fits"
    assert files[1]["Br"] == "br_2.fits"
    assert files[1]["Br_err"] == "br_err_2.fits"


def test_map_error_component_series_merge_into_file_series():
    config = normalize_config(
        _spherical_config(
            {
                "id": "full_disk",
                "type": "map",
                "files": [
                    {"Br": "br_1.fits", "Bt": "bt_1.fits", "Bp": "bp_1.fits"},
                    {"Br": "br_2.fits", "Bt": "bt_2.fits", "Bp": "bp_2.fits"},
                ],
                "errors": {
                    "Br_err": ["br_err_1.fits", "br_err_2.fits"],
                    "Bt_err": ["bt_err_1.fits", "bt_err_2.fits"],
                    "Bp_err": ["bp_err_1.fits", "bp_err_2.fits"],
                },
            }
        )
    )

    files = config["data"]["boundaries"][0]["files"]
    assert files[0]["Br_err"] == "br_err_1.fits"
    assert files[1]["Br_err"] == "br_err_2.fits"


def test_map_error_series_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="errors.*length"):
        normalize_config(
            _spherical_config(
                {
                    "id": "full_disk",
                    "type": "map",
                    "files": [
                        {"Br": "br_1.fits", "Bt": "bt_1.fits", "Bp": "bp_1.fits"},
                        {"Br": "br_2.fits", "Bt": "bt_2.fits", "Bp": "bp_2.fits"},
                    ],
                    "errors": [
                        {"Br_err": "br_err_1.fits", "Bt_err": "bt_err_1.fits", "Bp_err": "bp_err_1.fits"},
                        {"Br_err": "br_err_2.fits", "Bt_err": "bt_err_2.fits", "Bp_err": "bp_err_2.fits"},
                        {"Br_err": "br_err_3.fits", "Bt_err": "bt_err_3.fits", "Bp_err": "bp_err_3.fits"},
                    ],
                }
            )
        )
