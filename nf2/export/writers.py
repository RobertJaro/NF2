from __future__ import annotations

import json
import os

import h5py
import numpy as np
from astropy.io import fits

from nf2.evaluation.vtk import save_vtk


def _value(array):
    return array.to_value() if hasattr(array, "to_value") else array


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_vtk(path, payload):
    _ensure_parent(path)
    save_vtk(
        path,
        coords=_value(payload["coords"]),
        vectors={k: _value(v) for k, v in payload["vectors"].items()},
        scalars={k: _value(v) for k, v in payload["scalars"].items()},
    )


def write_hdf5(path, payload):
    _ensure_parent(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=_value(payload["coords"]), dtype="f4", compression="gzip")
        vectors = f.create_group("vectors")
        for key, value in payload["vectors"].items():
            vectors.create_dataset(key, data=_value(value), dtype="f4", compression="gzip")
        scalars = f.create_group("scalars")
        for key, value in payload["scalars"].items():
            scalars.create_dataset(key, data=_value(value), dtype="f4", compression="gzip")
        for key, value in payload["metadata"].items():
            f.attrs[key] = value


def write_npz(path, payload):
    _ensure_parent(path)
    save_dict = {"coords": _value(payload["coords"])}
    save_dict.update({f"vector__{k}": _value(v) for k, v in payload["vectors"].items()})
    save_dict.update({f"scalar__{k}": _value(v) for k, v in payload["scalars"].items()})
    save_dict["metadata"] = np.array(json.dumps(payload["metadata"]))
    np.savez(path, **save_dict)


def write_binary(path, payload):
    os.makedirs(path, exist_ok=True)
    b_field = payload["vectors"]["b"]
    _value(b_field).tofile(os.path.join(path, "B.bin"))
    with open(os.path.join(path, "README.txt"), "w") as f:
        f.write(f"Geometry: {payload['geometry']}\n")
        f.write(f"B shape: {_value(b_field).shape}\n")
        f.write(f"Coords shape: {_value(payload['coords']).shape}\n")
        for key, value in payload["metadata"].items():
            f.write(f"{key}: {value}\n")


def write_fits(path, payload):
    _ensure_parent(path)
    header = fits.Header()
    for key, value in payload["metadata"].items():
        hkey = key.upper().replace("_", "")[:8]
        try:
            header[hkey] = value
        except Exception:
            header[hkey] = str(value)

    hdus = [fits.PrimaryHDU(_value(payload["vectors"]["b"]), header=header)]
    for key, value in payload["vectors"].items():
        if key == "b":
            continue
        hdus.append(fits.ImageHDU(_value(value), name=f"V_{key.upper()[:6]}"))
    for key, value in payload["scalars"].items():
        hdus.append(fits.ImageHDU(_value(value), name=f"S_{key.upper()[:6]}"))
    hdus.append(fits.ImageHDU(_value(payload["coords"]), name="COORDS"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


WRITERS = {
    "vtk": write_vtk,
    "hdf5": write_hdf5,
    "npz": write_npz,
    "binary": write_binary,
    "fits": write_fits,
}
