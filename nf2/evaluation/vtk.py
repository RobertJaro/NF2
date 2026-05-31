import os

import numpy as np


def _as_array(value):
    if hasattr(value, "to_value"):
        value = value.to_value()
    return np.asarray(value)


def _vtk_name(name):
    return str(name).replace(" ", "_")


def split_vectors_scalars(values):
    """Split VTK-ready arrays into vector and scalar dictionaries by shape."""
    vectors = {}
    scalars = {}
    for name, value in values.items():
        shape = value.shape
        if len(shape) == 4 and shape[-1] == 3:
            vectors[name] = value
        elif len(shape) == 3:
            scalars[name] = value
    return vectors, scalars


def save_vtk(path, coords=None, vectors=None, scalars=None, Mm_per_pix=720e-3):
    """Save a structured grid as a legacy ASCII VTK file.

    This lightweight writer intentionally avoids the old Mayavi/TVTK dependency.
    Arrays are expected in NF2 order ``(x, y, z[, component])`` and are written
    in the point order used by VTK structured grids.
    """
    vectors = vectors if vectors is not None else {}
    scalars = scalars if scalars is not None else {}

    if len(vectors) > 0:
        dim = _as_array(next(iter(vectors.values()))).shape[:-1]
    elif len(scalars) > 0:
        dim = _as_array(next(iter(scalars.values()))).shape
    else:
        raise ValueError("No data to save.")

    if coords is None:
        pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.float32) * Mm_per_pix
        pts[..., 0] -= dim[0] * Mm_per_pix / 2
        pts[..., 1] -= dim[1] * Mm_per_pix / 2
    else:
        pts = _as_array(coords).astype(np.float32)

    points = pts.transpose(2, 1, 0, 3).reshape((-1, 3))
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("NF2 structured grid\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {dim[0]} {dim[1]} {dim[2]}\n")
        f.write(f"POINTS {points.shape[0]} float\n")
        np.savetxt(f, points, fmt="%.8e")
        f.write(f"\nPOINT_DATA {points.shape[0]}\n")

        for name, values in vectors.items():
            values = _as_array(values).astype(np.float32)
            expected = tuple(dim) + (3,)
            if values.shape != expected:
                raise ValueError(f"Vector {name!r} has incompatible shape {values.shape}; expected {expected}.")
            flat = values.transpose(2, 1, 0, 3).reshape((-1, 3))
            f.write(f"VECTORS {_vtk_name(name)} float\n")
            np.savetxt(f, flat, fmt="%.8e")
            f.write("\n")

        for name, values in scalars.items():
            values = _as_array(values).astype(np.float32)
            if values.shape != tuple(dim):
                raise ValueError(f"Scalar {name!r} has incompatible shape {values.shape}; expected {dim}.")
            flat = values.transpose(2, 1, 0).reshape((-1,))
            f.write(f"SCALARS {_vtk_name(name)} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, flat, fmt="%.8e")
            f.write("\n")
