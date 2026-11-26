"""Command-line quality metrics for NF2 extrapolation results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from astropy import units as u

from nf2.evaluation.metric_catalog import GROUPS, METRICS, select_metrics


def _to_value(value, unit=None):
    if hasattr(value, "to_value"):
        return value.to_value(unit) if unit is not None else value.value
    return value


def _quality_metrics(b, jac_matrix, names):
    """Compute the selected scalar reductions using the neural-field Jacobian."""
    from nf2.train.model import calculate_current_from_jacobian

    results = {}
    b = _to_value(b, u.G)
    jac = _to_value(jac_matrix, u.G / u.Mm)
    eps = 1e-12
    b_norm = np.linalg.norm(b, axis=-1)
    if any("divB" in name for name in names):
        div = np.trace(jac, axis1=-2, axis2=-1)
        for name in names:
            if "divB" in name:
                residual = div / (b_norm + eps) if name.endswith("over_B") else div
                results[name] = _reduce_residual(name, residual)
    if any("divB" not in name for name in names):
        j = calculate_current_from_jacobian(jac, f=np)
        cross = np.linalg.norm(np.cross(j, b), axis=-1)
        if "sigma_J" in names or "theta_J" in names:
            j_norm = np.linalg.norm(j, axis=-1)
            sine = np.clip(cross / (j_norm * b_norm + eps), 0, 1)
            sigma = np.nansum(sine * (j_norm / (np.nansum(j_norm) + eps)))
            results.update(sigma_J=sigma, theta_J=np.rad2deg(np.arcsin(np.clip(sigma, 0, 1))))
        for name in names:
            if name.endswith("force_free_residual"):
                results[name] = _reduce_residual(name, cross / (b_norm + eps))
    return results


def _reduce_residual(name, residual):
    if name.startswith("rms"):
        return np.sqrt(np.nanmean(residual ** 2))
    return np.nanmean(np.abs(residual))


def _cartesian_volume(output):
    mm_per_pixel = output["Mm_per_pixel"]
    return ((mm_per_pixel * u.Mm).to(u.cm)) ** 3


def _spherical_volume(output):
    spherical = output["spherical_coords"]
    r = spherical[..., 0] * u.solRad
    theta = spherical[..., 1]
    phi = spherical[..., 2]
    dr = np.nanmean(np.gradient(r.to_value(u.cm), axis=0)) * u.cm
    dtheta = np.nanmean(np.gradient(theta, axis=1)) * u.dimensionless_unscaled
    dphi = np.nanmean(np.gradient(phi, axis=2)) * u.dimensionless_unscaled
    return (r.to(u.cm) ** 2 * np.sin(theta) * dr * dtheta * dphi).to(u.cm ** 3)


def _energy_density(b):
    b_gauss = _to_value(b, u.G)
    return (np.sum(b_gauss ** 2, axis=-1) / (8 * np.pi)) * u.erg / u.cm ** 3


def _energy_metrics(output, geometry, names):
    b = output["b"]
    volume = _cartesian_volume(output) if geometry == "cartesian" else _spherical_volume(output)
    e_density = _energy_density(b)
    e_tot = np.nansum((e_density * volume).to_value(u.erg)) * u.erg
    metrics = {"E_tot": e_tot.to_value(u.erg)}

    if names.difference({"E_tot"}):
        from nf2.evaluation.energy import get_free_mag_energy

        free_density = get_free_mag_energy(_to_value(b, u.G), method="fft") * u.erg / u.cm ** 3
        e_free = np.nansum((free_density * volume).to_value(u.erg)) * u.erg
        metrics["E_free"] = e_free.to_value(u.erg)
        metrics["E_pot"] = (e_tot - e_free).to_value(u.erg)
        metrics["E_free_over_E_tot"] = (e_free / e_tot).to_value(u.dimensionless_unscaled) if e_tot != 0 else math.nan
    return metrics


def _regular_grid(out, geometry, options, sample_args):
    if geometry == "cartesian":
        xy = np.asarray(out.coord_range[:2]) * out.Mm_per_ds
        bounds = [options.get("x_range", xy[0]), options.get("y_range", xy[1]),
                  options.get("height_range", (0., out.max_height))]
        spacing = options.get("Mm_per_pixel", out.Mm_per_pixel)
        if np.any(np.round(np.diff(bounds, axis=1) / spacing) < 1):
            raise ValueError("Cartesian grid is empty. Reduce --Mm_per_pixel or increase the sampled extent.")
        output = out.load_cube(**options, **sample_args)
        axes = ["x_Mm", "y_Mm", "z_Mm"]
        coords = output["coords"]
    else:
        options = options.copy()
        for key, unit in (("radius_range", u.solRad), ("latitude_range", u.deg), ("longitude_range", u.deg)):
            if key in options:
                options[key] = np.asarray(options[key]) * unit
        output = out.load_spherical(**options, **sample_args)
        axes = ["radius_solRad", "colatitude_rad", "longitude_rad"]
        coords = output["spherical_coords"]
    # Both standard grids are separable and ordered monotonically along each axis.
    first, last = coords[0, 0, 0], coords[-1, -1, -1]
    grid = {"shape": list(coords.shape[:3]),
            "axes": {axis: [float(start), float(end)] for axis, start, end in zip(axes, first, last)}}
    if geometry == "cartesian":
        grid["spacing_Mm"] = float(output["Mm_per_pixel"])
    return output, grid


def _evaluate(nf2_path, metrics=None, *, device=None, progress=False, batch_size=None,
              Mm_per_pixel=None, height_range=None, x_range=None, y_range=None,
              spherical_sampling=None, radius_range=None, latitude_range=None, longitude_range=None):
    import nf2
    from nf2.evaluation.helicity import evaluate_box

    if metrics is not None and not isinstance(metrics, str):
        metrics = tuple(metrics)
    select_metrics(metrics)  # Catch typos before loading a potentially large checkpoint.
    out = nf2.load(nf2_path, device=device)
    geometry = out.state["data"]["type"]
    names = select_metrics(metrics, geometry)
    selected = set(names)
    cartesian = {"Mm_per_pixel": Mm_per_pixel, "height_range": height_range, "x_range": x_range, "y_range": y_range}
    spherical = {"sampling": spherical_sampling, "radius_range": radius_range,
                 "latitude_range": latitude_range, "longitude_range": longitude_range}
    options, incompatible = (cartesian, spherical) if geometry == "cartesian" else (spherical, cartesian)
    for key, value in incompatible.items():
        if value is not None:
            raise ValueError(f"{key} does not apply to {geometry} checkpoints.")
    options = {k: v for k, v in options.items() if v is not None}
    for key, value in options.items():
        if key.endswith("_range") and (len(value) != 2 or not np.isfinite(value).all() or value[1] <= value[0]):
            raise ValueError(f"{key} requires two increasing finite bounds.")
    if Mm_per_pixel is not None and (not np.isfinite(Mm_per_pixel) or Mm_per_pixel <= 0):
        raise ValueError("Mm_per_pixel must be positive and finite.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if spherical_sampling is not None and (len(spherical_sampling) != 3 or min(spherical_sampling) < 2):
        raise ValueError("spherical_sampling requires three sizes of at least two.")
    quality = selected.intersection(GROUPS["quality"])
    energies = selected.intersection(GROUPS["energies"])
    strength = selected.intersection(GROUPS["strength"])
    values, grids = {}, {}
    if quality or energies or strength:
        sample_args = {"compute_jacobian": bool(quality), "progress": progress}
        if batch_size is not None:
            sample_args["batch_size"] = batch_size
        if geometry == "spherical":
            options.setdefault("sampling", (32, 64, 128))
        elif "height_range" not in options and "z_range" in out.data_config:
            options["height_range"] = out.data_config["z_range"]
        output, grids["standard"] = _regular_grid(out, geometry, options, sample_args)
        if quality:
            values.update(_quality_metrics(output["b"], output["jac_matrix"], quality))
        if strength:
            norm = np.linalg.norm(output["b"].to_value(u.G), axis=-1)
            if "mean_B" in strength:
                values["mean_B"] = np.nanmean(norm)
            if "max_B" in strength:
                values["max_B"] = np.nanmax(norm)
        if energies:
            values.update(_energy_metrics(output, geometry, energies))
        del output

    closed = selected.difference(quality, energies, strength)
    if closed:
        faces = [face for face in ("bottom", "top") if any(name.endswith("_" + face) for name in closed)]
        box = evaluate_box(out, Mm_per_pixel, batch_size or 4096, height_range=height_range, x_range=x_range,
                           y_range=y_range, directions=faces, progress=progress,
                           flux_only=not faces and "rms_div_B_G_per_Mm" not in closed)
        grids["closed_box"] = {k: box[k] for k in ("box_Mm", "grid_shape", "spacing_Mm")}
        values.update(box["diagnostics"])
        for face in faces:
            for key, value in box["from_" + face].items():
                values[key.removesuffix("_Mx2") + "_" + face] = value

    values = {name: float(values[name]) for name in names}
    return {"checkpoint": str(nf2_path), "geometry": geometry, "grids": grids}, values


def compute_metrics(nf2_path, *, metrics=None, **kwargs):
    """Return geometry and selected scalar diagnostics as a flat dictionary."""
    metadata, values = _evaluate(nf2_path, metrics, **kwargs)
    return {"geometry": metadata["geometry"], **values}


def _rows(values):
    for name, value in values.items():
        metric = METRICS[name]
        grid = "closed_box" if metric.group in ("flux", "helicity") else "standard"
        direction = name.rsplit("_", 1)[1] if name.startswith("H_") else None
        yield {"metric": name, "value": value if np.isfinite(value) else None,
               "unit": metric.unit, "grid": grid, "direction": direction}


def _table(values):
    lines = []
    for row in _rows(values):
        value = "n/a" if row["value"] is None else f'{row["value"]:.6e}'
        lines.append(f'{row["metric"]:>28}  {value:>14}  {row["unit"]}')
    return "\n".join(lines)


def print_metrics(metrics):
    """Print a flat compute_metrics result to the terminal."""
    values = {k: v for k, v in metrics.items() if k != "geometry"}
    print(f'NF2 quality metrics ({metrics["geometry"]})')
    print(_table(values))


def _report(stream, suffix, metadata, values):
    rows = list(_rows(values))
    if suffix == ".json":
        json.dump({**metadata, "metrics": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    elif suffix == ".csv":
        writer = csv.DictWriter(stream, fieldnames=["checkpoint", "geometry", "metric", "value", "unit", "direction", "grid", "sampling"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "checkpoint": metadata["checkpoint"], "geometry": metadata["geometry"],
                             "sampling": json.dumps(metadata["grids"][row["grid"]])})
    else:
        print(f'NF2 quality metrics ({metadata["geometry"]})  {metadata["checkpoint"]}', file=stream)
        for name, grid in metadata["grids"].items():
            print(name, json.dumps(grid), file=stream)
        print(_table(values), file=stream)


def _write_report(path, metadata, values):
    temp = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="", encoding="utf-8") as file:
            temp = Path(file.name)
            _report(file, path.suffix.lower(), metadata, values)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Evaluate scalar diagnostics for an NF2 checkpoint.")
    parser.add_argument("nf2_path", nargs="?", help="Path to extrapolation_result.nf2")
    parser.add_argument("--metrics", nargs="+", metavar="NAME", help="Metric names or groups (default: standard)")
    parser.add_argument("--list-metrics", action="store_true", help="Show supported metrics and groups")
    parser.add_argument("--output", type=Path, help="Write a .txt, .csv, or .json report")
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--Mm_per_pixel", type=float, default=None)
    parser.add_argument("--height_range", type=float, nargs=2, default=None)
    parser.add_argument("--x_range", type=float, nargs=2, default=None)
    parser.add_argument("--y_range", type=float, nargs=2, default=None)
    parser.add_argument("--spherical_sampling", type=int, nargs=3, default=None)
    parser.add_argument("--radius_range", type=float, nargs=2, default=None)
    parser.add_argument("--latitude_range", type=float, nargs=2, default=None)
    parser.add_argument("--longitude_range", type=float, nargs=2, default=None)
    args = vars(parser.parse_args(argv))
    if args.pop("list_metrics"):
        print("Groups:", ", ".join(GROUPS), "(default: standard = quality + energies)")
        for name, spec in METRICS.items():
            geometry = "Cartesian" if spec.cartesian_only else "Cartesian/spherical"
            print(f"{name:28} {spec.unit:8} {spec.group:10} {geometry:20} {spec.description}")
        return
    path = args.pop("nf2_path")
    if path is None:
        parser.error("A checkpoint is required unless using --list-metrics.")
    output = args.pop("output")
    if output and output.suffix.lower() not in (".txt", ".csv", ".json"):
        parser.error("--output must end in .txt, .csv or .json")
    try:
        metadata, values = _evaluate(path, **args)
        if output:
            _write_report(output, metadata, values)
        else:
            _report(sys.stdout, ".txt", metadata, values)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
