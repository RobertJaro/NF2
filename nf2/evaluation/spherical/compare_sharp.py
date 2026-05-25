import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

from nf2.data.util import vector_cartesian_to_spherical
from nf2.evaluation.metric import divergence_jacobian, sigma_J, theta_J, vector_norm
from nf2.evaluation.output import CartesianOutput, SphericalOutput


DEFAULT_Mm_PER_PIXEL = 0.72
DEFAULT_HEIGHT = 20 * u.Mm


def _quantity_to_value(array, unit=None):
    if hasattr(array, "to_value"):
        return array.to_value(unit) if unit is not None else array.value
    return array


def _get_reference_coordinates(reference_map):
    carrington_coords = all_coordinates_from_map(reference_map).transform_to(frames.HeliographicCarrington)
    center_lon = reference_map.center.transform_to(frames.HeliographicCarrington).lon
    lon = carrington_coords.lon.wrap_at(center_lon + 180 * u.deg).to_value(u.deg)
    lat = carrington_coords.lat.to_value(u.deg)
    return lon, lat


def _extent_from_coordinates(lon, lat):
    return [
        np.nanmin(lon),
        np.nanmax(lon),
        np.nanmin(lat),
        np.nanmax(lat),
    ]


def _add_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", label=label)


def _plot_panel(fig, ax, data, extent, title, cmap, norm=None, vmin=None, vmax=None, label=None):
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if label is not None:
        _add_colorbar(fig, ax, im, label)
    return im


def _format_boundary_ticks(axs, extent):
    x_ticks = [extent[0], extent[1]]
    y_ticks = [extent[2], extent[3]]
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f"{tick:.1f}" for tick in x_ticks])
            ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks])
            ax.tick_params(labelbottom=i == len(axs) - 1, labelleft=j == 0)


def _integrate_current_density(j, axis, Mm_per_pixel):
    current_density = np.linalg.norm(_quantity_to_value(j, u.G / u.s), axis=-1)
    return np.nansum(current_density, axis=axis) * (Mm_per_pixel * u.Mm).to_value(u.cm)


def _load_aia_193(aia_path, reference_map):
    aia_map = Map(aia_path)
    exposure_time = aia_map.exposure_time.to_value(u.s) if hasattr(aia_map, "exposure_time") else 1
    aia_map = Map(aia_map.data / exposure_time, aia_map.meta)
    return aia_map.reproject_to(reference_map.wcs)


def _compute_box_metrics(model_out):
    b = _quantity_to_value(model_out["b"], u.G)
    j = _quantity_to_value(model_out["metrics"]["j"], u.G / u.s)
    jac_matrix = _quantity_to_value(model_out["jac_matrix"], u.G / u.Mm)
    div_b = divergence_jacobian(jac_matrix)
    norm_b = vector_norm(b)

    return {
        "theta_J [deg]": theta_J(b, j),
        "sigma_J": sigma_J(b, j),
        "mean |divB| [G/Mm]": np.nanmean(np.abs(div_b)),
        "mean |divB|/|B| [1/Mm]": np.nanmean(np.abs(div_b) / (norm_b + 1e-7)),
    }


def _write_metrics_report(metrics_out_path, metrics):
    headers = ["Model", *next(iter(metrics.values())).keys()]
    rows = [[name, *values.values()] for name, values in metrics.items()]
    widths = [
        max(len(headers[i]), *(len(f"{row[i]:.6e}") if isinstance(row[i], float) else len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]

    lines = ["NF2 SHARP comparison metrics", "=" * 28, ""]
    lines.append("Currents use the representation Jacobian; divB uses the model Jacobian trace.")
    lines.append("")
    lines.append("  ".join(f"{header:<{widths[i]}}" if i == 0 else f"{header:>{widths[i]}}"
                           for i, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(
            f"{value:<{widths[i]}}" if i == 0 else f"{value:>{widths[i]}.6e}"
            for i, value in enumerate(row)
        ))
    lines.append("")

    out_dir = os.path.dirname(metrics_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(metrics_out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare SHARP, cartesian, and spherical NF2 outputs.")
    parser.add_argument("--cartesian_nf2", type=str, required=True)
    parser.add_argument("--spherical_nf2", type=str, required=True)
    parser.add_argument("--reference_br", type=str, required=True)
    parser.add_argument("--aia_193", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--metrics_out_path", type=str, default=None)
    parser.add_argument("--Mm_per_pixel", type=float, default=DEFAULT_Mm_PER_PIXEL)
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT.to_value(u.Mm), help="Height in Mm.")
    args = parser.parse_args()

    reference_map = Map(args.reference_br)
    aia_193_map = _load_aia_193(args.aia_193, reference_map)
    carrington_lon, carrington_lat = _get_reference_coordinates(reference_map)
    extent = _extent_from_coordinates(carrington_lon, carrington_lat)
    longitude_range = (np.nanmin(carrington_lon), np.nanmax(carrington_lon)) * u.deg
    latitude_range = (np.nanmin(carrington_lat), np.nanmax(carrington_lat)) * u.deg

    cartesian_model = CartesianOutput(args.cartesian_nf2)
    cartesian_out = cartesian_model.load_cube(
        height_range=(0, args.height),
        Mm_per_pixel=args.Mm_per_pixel,
        metrics=["j"],
        progress=True,
    )

    sampling = [
        cartesian_out["coords"].shape[2],
        cartesian_out["coords"].shape[1],
        cartesian_out["coords"].shape[0],
    ]
    spherical_model = SphericalOutput(args.spherical_nf2)
    spherical_out = spherical_model.load_spherical(
        latitude_range=latitude_range,
        longitude_range=longitude_range,
        radius_range=(1.0 * u.solRad, 1.0 * u.solRad + args.height * u.Mm),
        sampling=sampling,
        metrics=["j"],
        progress=True,
    )
    metrics_out_path = args.metrics_out_path
    if metrics_out_path is None:
        out_root, _ = os.path.splitext(args.out_path)
        metrics_out_path = f"{out_root}_metrics.txt"
    _write_metrics_report(metrics_out_path, {
        "Cartesian": _compute_box_metrics(cartesian_out),
        "Spherical": _compute_box_metrics(spherical_out),
    })

    reference_br = reference_map.data
    cartesian_br = _quantity_to_value(cartesian_out["b"], u.G)[:, :, 0, 2].T
    spherical_b = _quantity_to_value(spherical_out["b"], u.G)
    spherical_br = np.flipud(vector_cartesian_to_spherical(spherical_b, spherical_out["spherical_coords"])[0, :, :, 0])

    cartesian_currents = _integrate_current_density(cartesian_out["metrics"]["j"], axis=2,
                                                    Mm_per_pixel=args.Mm_per_pixel).T
    spherical_currents = np.flipud(_integrate_current_density(spherical_out["metrics"]["j"], axis=0,
                                                              Mm_per_pixel=args.Mm_per_pixel))

    positive_currents = cartesian_currents[np.isfinite(cartesian_currents) & (cartesian_currents > 0)]
    if positive_currents.size > 0:
        current_vmin = np.nanmin(positive_currents)
        current_vmax = np.nanmax(positive_currents)
        current_vmax = current_vmax if current_vmax > current_vmin else current_vmin * 1.01
        current_norm = LogNorm(vmin=current_vmin, vmax=current_vmax)
    else:
        current_norm = None
    positive_aia = aia_193_map.data[np.isfinite(aia_193_map.data) & (aia_193_map.data > 0)]
    if positive_aia.size > 0:
        aia_vmin = max(np.nanpercentile(positive_aia, 1), 1e-3)
        aia_vmax = np.nanpercentile(positive_aia, 99.5)
        aia_vmax = aia_vmax if aia_vmax > aia_vmin else aia_vmin * 1.01
        aia_norm = LogNorm(vmin=aia_vmin, vmax=aia_vmax)
    else:
        aia_norm = None
    br_norm = Normalize(vmin=-1000, vmax=1000)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)

    _plot_panel(fig, axs[0, 0], reference_br, extent, "Br SHARP", "gray", norm=br_norm, label="Br [G]")
    _plot_panel(fig, axs[0, 1], cartesian_br, extent, "Br Cartesian", "gray", norm=br_norm, label="Br [G]")
    _plot_panel(fig, axs[0, 2], spherical_br, extent, "Br Spherical", "gray", norm=br_norm, label="Br [G]")

    _plot_panel(fig, axs[1, 0], aia_193_map.data, extent, r"AIA 193 $\AA$", cm.sdoaia193,
                norm=aia_norm, label=r"AIA 193 $\AA$ [DN/s]")
    _plot_panel(fig, axs[1, 1], cartesian_currents, extent, "Currents Cartesian", "inferno",
                norm=current_norm, label=r"$\int |J|\,dh$ [G cm s$^{-1}$]")
    _plot_panel(fig, axs[1, 2], spherical_currents, extent, "Currents Spherical", "inferno",
                norm=current_norm, label=r"$\int |J|\,dh$ [G cm s$^{-1}$]")

    _format_boundary_ticks(axs, extent)
    fig.supxlabel("Carrington Longitude [deg]")
    fig.supylabel("Carrington Latitude [deg]")
    fig.tight_layout()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_path, dpi=300, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
