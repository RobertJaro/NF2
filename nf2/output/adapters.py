from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u

from nf2.evaluation.output import CartesianOutput, SphericalOutput


def _quantity_to_value(value):
    return value.value if hasattr(value, "unit") else value


def _serialize_metadata(metadata):
    serialized = {}
    for key, value in metadata.items():
        if value is None:
            serialized[key] = "None"
        elif isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


class BaseOutputAdapter(ABC):
    geometry = None

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.output = self._build_output(checkpoint_path, device=device)

    @abstractmethod
    def _build_output(self, checkpoint_path, device=None):
        raise NotImplementedError

    @abstractmethod
    def sample(self, **kwargs):
        raise NotImplementedError

    def _base_metadata(self):
        data_config = getattr(self.output, "data_config", self.output.state["data"])
        return {
            "geometry": self.geometry,
            "checkpoint_path": self.checkpoint_path,
            "data_type": data_config["type"],
        }


class CartesianOutputAdapter(BaseOutputAdapter):
    geometry = "cartesian"

    def _build_output(self, checkpoint_path, device=None):
        return CartesianOutput(checkpoint_path, device=device)

    def sample(
        self,
        Mm_per_pixel=None,
        height_range=None,
        x_range=None,
        y_range=None,
        metrics=None,
        progress=False,
        **kwargs,
    ):
        result = self.output.load_cube(
            Mm_per_pixel=Mm_per_pixel,
            height_range=height_range,
            x_range=x_range,
            y_range=y_range,
            metrics=metrics,
            progress=progress,
            **kwargs,
        )

        vectors = {"b": result["b"]}
        scalars = {}
        for key, value in result.get("metrics", {}).items():
            if getattr(value, "ndim", None) == 4 and value.shape[-1] == 3:
                vectors[key] = value
            elif getattr(value, "ndim", None) == 3:
                scalars[key] = value

        metadata = self._base_metadata()
        metadata.update(
            {
                "Mm_per_pixel": result["Mm_per_pixel"],
                "wcs": self.output.wcs[0].to_header_string() if self.output.wcs and len(self.output.wcs) > 0 else "None",
                "time": self.output.time.isoformat("T", timespec="seconds") if self.output.time is not None else "None",
            }
        )

        return {
            "geometry": self.geometry,
            "coords": result["coords"],
            "vectors": vectors,
            "scalars": scalars,
            "metadata": _serialize_metadata(metadata),
        }


class SphericalOutputAdapter(BaseOutputAdapter):
    geometry = "spherical"

    def _build_output(self, checkpoint_path, device=None):
        return SphericalOutput(checkpoint_path, device=device)

    def sample(
        self,
        radius_range=None,
        latitude_range=None,
        longitude_range=None,
        pixels_per_solRad=None,
        metrics=None,
        progress=False,
        **kwargs,
    ):
        radius_range = self.output.radius_range if radius_range is None else tuple(radius_range) * u.solRad
        latitude_range = (-90, 90) if latitude_range is None else latitude_range
        longitude_range = (0, 360) if longitude_range is None else longitude_range
        pixels_per_solRad = 64 if pixels_per_solRad is None else pixels_per_solRad

        if not hasattr(latitude_range, "unit"):
            latitude_range = tuple(latitude_range) * u.deg
        if not hasattr(longitude_range, "unit"):
            longitude_range = tuple(longitude_range) * u.deg
        if not hasattr(pixels_per_solRad, "unit"):
            pixels_per_solRad = pixels_per_solRad * u.pix / u.solRad

        result = self.output.load(
            radius_range=radius_range,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            resolution=pixels_per_solRad,
            metrics=metrics,
            progress=progress,
            **kwargs,
        )

        vectors = {"b": result["b"], "b_rtp": result["b_rtp"]}
        scalars = {"radius": result["spherical_coords"][..., 0]}
        for key, value in result.get("metrics", {}).items():
            if getattr(value, "ndim", None) == 4 and value.shape[-1] == 3:
                vectors[key] = value
                scalars[f"{key}_magnitude"] = np.linalg.norm(_quantity_to_value(value), axis=-1)
            elif getattr(value, "ndim", None) == 3:
                scalars[key] = value

        metadata = self._base_metadata()
        metadata.update(
            {
                "radius_range": tuple(_quantity_to_value(v) for v in radius_range),
                "latitude_range_deg": tuple(latitude_range.to_value(u.deg)),
                "longitude_range_deg": tuple(longitude_range.to_value(u.deg)),
                "pixels_per_solRad": pixels_per_solRad.to_value(u.pix / u.solRad),
            }
        )

        return {
            "geometry": self.geometry,
            "coords": result["coords"],
            "vectors": vectors,
            "scalars": scalars,
            "metadata": _serialize_metadata(metadata),
        }
