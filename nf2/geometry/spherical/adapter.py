from __future__ import annotations

from copy import deepcopy

from nf2.core.adapters import GeometryAdapter, register_geometry_adapter


@register_geometry_adapter
class SphericalGeometryAdapter(GeometryAdapter):
    geometry = "spherical"

    _DATASET_TYPES = {
        "spherical",
        "map",
        "pfss_boundary",
        "random_spherical",
        "random_radial_grouped",
        "sphere",
        "spherical_slices",
    }

    @classmethod
    def matches(cls, data_config, series=False):
        data_type = data_config.get("type")
        if data_type == "spherical":
            return True
        if "synoptic_fits_path" in data_config or "fits_paths" in data_config:
            return True
        if "train_configs" in data_config:
            train_configs = data_config["train_configs"]
            if not train_configs:
                return False
            return any(config.get("type") in cls._DATASET_TYPES for config in train_configs)
        return False

    def prepare_data_config(self, data_config, series=False):
        config = deepcopy(data_config)
        config.pop("type", None)
        if series and "fits_paths" not in config and "data_path" in config:
            config["fits_paths"] = config.pop("data_path")
        if "validation_configs" not in config:
            config["validation_configs"] = []
        return config

    def create_data_module(self, data_config):
        from nf2.loader.spherical import SphericalDataModule

        return SphericalDataModule(**data_config)

    def create_series_data_module(self, data_config, current_step):
        from nf2.loader.spherical import SphericalSeriesDataModule

        config = deepcopy(data_config)
        if current_step > 0:
            config["fits_paths"] = config["fits_paths"][current_step:]
        return SphericalSeriesDataModule(**config)
