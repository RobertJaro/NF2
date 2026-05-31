import pytest

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
