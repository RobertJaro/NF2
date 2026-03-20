from __future__ import annotations

from copy import deepcopy

import numpy as np

from nf2.data.analytical_field import get_analytic_b_field
from nf2.loader.base import MapDataset


DEFAULT_ANALYTICAL_CASES = {
    1: {
        "n": 1,
        "m": 1,
        "l": 0.3,
        "psi": np.pi / 4,
        "resolution": [64, 64, 64],
        "bounds": [-1, 1, -1, 1, 0, 2],
    },
    2: {
        "n": 1,
        "m": 1,
        "l": 0.3,
        "psi": np.pi * 0.15,
        "resolution": [80, 80, 72],
        "bounds": [-1, 1, -1, 1, 0, 2],
    },
}


def resolve_analytical_case_config(case=1, **overrides):
    if case not in DEFAULT_ANALYTICAL_CASES:
        available = sorted(DEFAULT_ANALYTICAL_CASES)
        raise ValueError(f"Invalid analytical case {case}. Available cases: {available}")

    config = deepcopy(DEFAULT_ANALYTICAL_CASES[case])
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def load_analytical_field(case=1, **overrides):
    config = resolve_analytical_case_config(case=case, **overrides)
    return get_analytic_b_field(**config)


class AnalyticalDataset(MapDataset):
    def __init__(
        self,
        case=1,
        resolution=None,
        bounds=None,
        n=None,
        m=None,
        l=None,
        psi=None,
        tau_surfaces=None,
        z_index=0,
        field_scale=None,
        plot=False,
        **kwargs,
    ):
        config = resolve_analytical_case_config(
            case=case,
            resolution=resolution,
            bounds=bounds,
            n=n,
            m=m,
            l=l,
            psi=psi,
        )
        config["tau_surfaces"] = tau_surfaces

        b_cube = get_analytic_b_field(**config)
        z_count = b_cube.shape[2]
        if z_index < 0:
            z_index = z_count + z_index
        if z_index < 0 or z_index >= z_count:
            raise ValueError(f"z_index {z_index} is out of bounds for analytical cube depth {z_count}")

        resolution = config["resolution"]
        bounds = config["bounds"]
        x = np.linspace(bounds[0], bounds[1], resolution[0], dtype=np.float32)
        y = np.linspace(bounds[2], bounds[3], resolution[1], dtype=np.float32)
        z = np.linspace(bounds[4], bounds[5], resolution[2], dtype=np.float32)
        coords = np.stack(np.meshgrid(x, y, np.array([z[z_index]], dtype=np.float32), indexing="ij"), -1)[:, :, 0]

        if field_scale is None:
            field_scale = kwargs.get("G_per_dB", 1.0)

        dx = (bounds[1] - bounds[0]) / max(resolution[0] - 1, 1)
        b = b_cube[:, :, z_index, :] * field_scale

        super().__init__(b=b, coords=coords, Mm_per_pixel=dx, plot=plot, **kwargs)
