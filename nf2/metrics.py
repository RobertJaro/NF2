"""Command-line quality metrics for NF2 extrapolation results."""

from __future__ import annotations

import argparse
import math

import numpy as np
from astropy import units as u


def _to_value(value, unit=None):
    if hasattr(value, "to_value"):
        return value.to_value(unit) if unit is not None else value.value
    return value


def _curl_from_jacobian(jac_matrix):
    dBx_dx = jac_matrix[..., 0, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dx = jac_matrix[..., 1, 0]
    dBy_dy = jac_matrix[..., 1, 1]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dx = jac_matrix[..., 2, 0]
    dBz_dy = jac_matrix[..., 2, 1]
    dBz_dz = jac_matrix[..., 2, 2]
    curl = np.stack([
        _to_value(dBz_dy - dBy_dz, u.G / u.Mm),
        _to_value(dBx_dz - dBz_dx, u.G / u.Mm),
        _to_value(dBy_dx - dBx_dy, u.G / u.Mm),
    ], axis=-1)
    div = _to_value(dBx_dx + dBy_dy + dBz_dz, u.G / u.Mm)
    return curl, div


def _field_quality_metrics(b, jac_matrix):
    b_gauss = _to_value(b, u.G)
    j, div_b = _curl_from_jacobian(jac_matrix)

    b_norm = np.linalg.norm(b_gauss, axis=-1)
    j_norm = np.linalg.norm(j, axis=-1)
    j_cross_b = np.cross(j, b_gauss, axis=-1)
    j_cross_b_norm = np.linalg.norm(j_cross_b, axis=-1)

    eps = 1e-12
    sigma = j_cross_b_norm / ((j_norm * b_norm) + eps)
    sigma = np.clip(sigma, 0, 1)
    current_weight = j_norm / (np.nansum(j_norm) + eps)
    sigma_j = float(np.nansum(sigma * current_weight))
    theta_j = float(np.rad2deg(np.arcsin(np.clip(sigma_j, 0, 1))))

    div_b_over_b = np.abs(div_b) / (b_norm + eps)
    force_free_residual = j_cross_b_norm / (b_norm + eps)

    return {
        "mean_abs_divB": float(np.nanmean(np.abs(div_b))),
        "rms_divB": float(np.sqrt(np.nanmean(div_b ** 2))),
        "mean_abs_divB_over_B": float(np.nanmean(div_b_over_b)),
        "rms_divB_over_B": float(np.sqrt(np.nanmean(div_b_over_b ** 2))),
        "theta_J": theta_j,
        "sigma_J": sigma_j,
        "CWsin": sigma_j,
        "mean_force_free_residual": float(np.nanmean(force_free_residual)),
        "rms_force_free_residual": float(np.sqrt(np.nanmean(force_free_residual ** 2))),
        "mean_B": float(np.nanmean(b_norm)),
        "max_B": float(np.nanmax(b_norm)),
    }


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


def _energy_metrics(output, geometry):
    b = output["b"]
    volume = _cartesian_volume(output) if geometry == "cartesian" else _spherical_volume(output)
    e_density = _energy_density(b)
    e_tot = np.nansum((e_density * volume).to_value(u.erg)) * u.erg
    metrics = {"E_tot": e_tot.to_value(u.erg)}

    if geometry == "cartesian":
        from nf2.evaluation.energy import get_free_mag_energy

        free_density = get_free_mag_energy(_to_value(b, u.G), method="fft") * u.erg / u.cm ** 3
        e_free = np.nansum((free_density * volume).to_value(u.erg)) * u.erg
        metrics["E_free"] = e_free.to_value(u.erg)
        metrics["E_pot"] = (e_tot - e_free).to_value(u.erg)
        metrics["E_free_over_E_tot"] = (e_free / e_tot).to_value(u.dimensionless_unscaled) if e_tot != 0 else math.nan
    else:
        metrics["E_free"] = math.nan
        metrics["E_pot"] = math.nan
        metrics["E_free_over_E_tot"] = math.nan
    return metrics


def compute_metrics(nf2_path, *, device=None, progress=False, **kwargs):
    """Compute standard NLFF quality metrics for an NF2 result.

    Parameters
    ----------
    nf2_path:
        Path to an ``extrapolation_result.nf2`` file.
    device:
        Optional PyTorch device for evaluation.
    progress:
        Show progress while sampling the model.
    **kwargs:
        Geometry-specific sampling options. Cartesian output accepts
        ``Mm_per_pixel``, ``height_range``, ``x_range``, and ``y_range``.
        Spherical output accepts ``spherical_sampling``, ``radius_range``,
        ``latitude_range``, and ``longitude_range``.

    Returns
    -------
    dict
        Metric names and scalar values ready for printing or downstream use.
    """
    import nf2

    out = nf2.load(nf2_path, device=device)
    geometry = out.state["data"]["type"]

    if geometry == "cartesian":
        load_kwargs = {
            "Mm_per_pixel": kwargs.get("Mm_per_pixel"),
            "height_range": kwargs.get("height_range"),
            "x_range": kwargs.get("x_range"),
            "y_range": kwargs.get("y_range"),
            "batch_size": kwargs.get("batch_size"),
            "progress": progress,
            "compute_jacobian": True,
        }
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        output = out.load_cube(**load_kwargs)
    elif geometry == "spherical":
        load_kwargs = {
            "sampling": kwargs.get("spherical_sampling"),
            "batch_size": kwargs.get("batch_size"),
            "progress": progress,
            "compute_jacobian": True,
        }
        if kwargs.get("radius_range") is not None:
            load_kwargs["radius_range"] = kwargs["radius_range"] * u.solRad
        if kwargs.get("latitude_range") is not None:
            load_kwargs["latitude_range"] = kwargs["latitude_range"] * u.deg
        if kwargs.get("longitude_range") is not None:
            load_kwargs["longitude_range"] = kwargs["longitude_range"] * u.deg
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        output = out.load_spherical(**load_kwargs)
    else:
        raise ValueError(f"Unsupported NF2 geometry: {geometry!r}")

    metrics = {
        "geometry": geometry,
        **_field_quality_metrics(output["b"], output["jac_matrix"]),
        **_energy_metrics(output, geometry),
    }
    return metrics


def _format_value(key, value):
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return "n/a"
    if key.startswith("E_") and key != "E_free_over_E_tot":
        return f"{value:.6e} erg"
    if key in {"mean_abs_divB", "rms_divB"}:
        return f"{value:.6e} G / Mm"
    if key.endswith("divB_over_B"):
        return f"{value:.6e} 1 / Mm"
    if key in {"mean_force_free_residual", "rms_force_free_residual"}:
        return f"{value:.6e} G / Mm"
    if key in {"theta_J"}:
        return f"{value:.6f} deg"
    if key in {"sigma_J", "CWsin", "E_free_over_E_tot"}:
        return f"{value:.6f}"
    if key in {"mean_B", "max_B"}:
        return f"{value:.6e} G"
    return f"{value:.6e}"


def print_metrics(metrics):
    """Print metrics in a compact command-line table."""
    order = [
        "geometry",
        "mean_abs_divB",
        "rms_divB",
        "mean_abs_divB_over_B",
        "rms_divB_over_B",
        "theta_J",
        "sigma_J",
        "CWsin",
        "mean_force_free_residual",
        "rms_force_free_residual",
        "E_tot",
        "E_free",
        "E_pot",
        "E_free_over_E_tot",
        "mean_B",
        "max_B",
    ]
    print("NF2 quality metrics")
    print("-------------------")
    for key in order:
        print(f"{key:>28}: {_format_value(key, metrics.get(key))}")
    if metrics.get("geometry") == "spherical":
        print("\nNote: E_free is n/a for spherical output because the current free-energy")
        print("      estimate uses the Cartesian FFT potential-field reference.")


def main():
    parser = argparse.ArgumentParser(description="Compute standard NLFF quality metrics for an NF2 result.")
    parser.add_argument("nf2_path", help="Path to an extrapolation_result.nf2 file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--Mm_per_pixel", type=float, default=None)
    parser.add_argument("--height_range", type=float, nargs=2, default=None)
    parser.add_argument("--x_range", type=float, nargs=2, default=None)
    parser.add_argument("--y_range", type=float, nargs=2, default=None)
    parser.add_argument("--spherical_sampling", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--radius_range", type=float, nargs=2, default=None)
    parser.add_argument("--latitude_range", type=float, nargs=2, default=None)
    parser.add_argument("--longitude_range", type=float, nargs=2, default=None)
    args = parser.parse_args()

    metrics = compute_metrics(
        args.nf2_path,
        device=args.device,
        progress=args.progress,
        batch_size=args.batch_size,
        Mm_per_pixel=args.Mm_per_pixel,
        height_range=args.height_range,
        x_range=args.x_range,
        y_range=args.y_range,
        spherical_sampling=args.spherical_sampling,
        radius_range=args.radius_range,
        latitude_range=args.latitude_range,
        longitude_range=args.longitude_range,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
