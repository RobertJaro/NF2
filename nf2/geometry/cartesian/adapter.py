from __future__ import annotations

from copy import deepcopy

from nf2.core.adapters import GeometryAdapter, register_geometry_adapter


@register_geometry_adapter
class CartesianGeometryAdapter(GeometryAdapter):
    geometry = "cartesian"

    _DATASET_TYPES = {
        "cartesian",
        "fits",
        "sharp",
        "los_trv_azi",
        "los",
        "fld_inc_azi",
        "numpy",
        "muram_slice",
        "muram_cube",
        "muram_pressure",
        "analytical",
    }

    @classmethod
    def matches(cls, data_config, series=False):
        data_type = data_config.get("type")
        if data_type in cls._DATASET_TYPES:
            return True
        if "slices" in data_config or "data_path" in data_config:
            return True
        if "train_configs" in data_config:
            train_configs = data_config["train_configs"]
            if not train_configs:
                return False
            return all(config.get("type") in cls._DATASET_TYPES for config in train_configs)
        return False

    def prepare_data_config(self, data_config, series=False):
        config = deepcopy(data_config)
        data_type = config.pop("type", "cartesian")

        if "validation_configs" in config and "valid_configs" not in config:
            config["valid_configs"] = config.pop("validation_configs")
        if "validation_ds_per_pixel" in config and "validation_pixel_per_ds" not in config:
            validation_ds_per_pixel = config.pop("validation_ds_per_pixel")
            if validation_ds_per_pixel:
                config["validation_pixel_per_ds"] = 1 / validation_ds_per_pixel

        if series:
            if "train_configs" not in config:
                source_type = "sharp" if data_type == "sharp" else "fits"
                if data_type == "muram_slice":
                    source_type = "muram_slice"
                config["train_configs"] = [{"type": source_type, "data_path": config.pop("data_path")}]
        else:
            if "train_configs" not in config and "slices" in config:
                config["train_configs"] = config.pop("slices")
        return config

    def create_data_module(self, data_config):
        from nf2.loader.cartesian import CartesianDataModule

        return CartesianDataModule(**data_config)

    def create_series_data_module(self, data_config, current_step):
        from nf2.loader.cartesian import CartesianSeriesDataModule

        return CartesianSeriesDataModule(current_step=current_step, **data_config)
