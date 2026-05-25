import argparse
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

from nf2.data.util import spherical_to_cartesian, vector_cartesian_to_spherical
from nf2.evaluation.output import SphericalOutput


DEFAULT_RADIUS_RANGE = (1.0, 1.2)
DEFAULT_SAMPLING = 128
DEFAULT_AIA_193_VMIN = 10
DEFAULT_AIA_193_VMAX = 5e3
DEFAULT_AIA_304_VMIN = 5
DEFAULT_AIA_304_VMAX = 1e3


def _quantity_to_value(array, unit=None):
    if hasattr(array, "to_value"):
        return array.to_value(unit) if unit is not None else array.value
    return array


def _load_aia(aia_path, hpc_x=None, hpc_y=None):
    aia_map = Map(aia_path)
    exposure_time = aia_map.exposure_time.to_value(u.s) if hasattr(aia_map, "exposure_time") else 1
    aia_map = Map(aia_map.data / exposure_time, aia_map.meta)

    aia_map = aia_map.resample((1024, 1024) * u.pixel)

    if hpc_x is None or hpc_y is None:
        return aia_map

    bottom_left = SkyCoord(hpc_x[0] * u.arcsec, hpc_y[0] * u.arcsec, frame=aia_map.coordinate_frame)
    top_right = SkyCoord(hpc_x[1] * u.arcsec, hpc_y[1] * u.arcsec, frame=aia_map.coordinate_frame)
    return aia_map.submap(bottom_left=bottom_left, top_right=top_right)


def _aia_colormap(channel):
    cmap_name = f"sdoaia{channel}"
    return getattr(cm, cmap_name, "gray")


def _extent_from_hpc(aia_map):
    return [
        aia_map.bottom_left_coord.Tx.to_value(u.arcsec),
        aia_map.top_right_coord.Tx.to_value(u.arcsec),
        aia_map.bottom_left_coord.Ty.to_value(u.arcsec),
        aia_map.top_right_coord.Ty.to_value(u.arcsec),
    ]


def _add_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.18)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", label=label)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")


def _plot_panel(fig, ax, data, extent, cmap, norm=None, label=None):
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, norm=norm)
    _add_colorbar(fig, ax, im, label)
    return im


def _log_norm_from_positive(data):
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        return None
    vmin = np.nanmin(positive)
    vmax = np.nanmax(positive)
    vmax = vmax if vmax > vmin else vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


def _load_spherical_projection(model, aia_map, radius_range, sampling, batch_size):
    carrington_coords = all_coordinates_from_map(aia_map).transform_to(frames.HeliographicCarrington)
    lon = carrington_coords.lon.to_value(u.rad)
    lat = carrington_coords.lat.to_value(u.rad)
    valid_mask = np.isfinite(lon) & np.isfinite(lat)

    br = np.full(aia_map.data.shape, np.nan)
    current_density = np.full(aia_map.data.shape, np.nan)
    if not np.any(valid_mask):
        return br, current_density

    radius = np.linspace(radius_range[0], radius_range[1], sampling)
    spherical_coords = np.stack(
        [
            np.repeat(radius[:, None], valid_mask.sum(), axis=1),
            np.repeat((np.pi / 2 - lat[valid_mask])[None], sampling, axis=0),
            np.repeat(lon[valid_mask][None], sampling, axis=0),
        ],
        axis=-1,
    )
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    scaled_coords = cartesian_coords * (1 * u.solRad / model.m_per_ds).to_value(u.dimensionless_unscaled)

    model_out = model.load_coords(
        scaled_coords,
        metrics=["j"],
        progress=True,
        batch_size=batch_size,
    )

    b_rtp = vector_cartesian_to_spherical(model_out["b"], spherical_coords)
    br[valid_mask] = _quantity_to_value(b_rtp[0, :, 0], u.G)

    radius_cm = (radius * u.solRad).to_value(u.cm)
    j_norm = np.linalg.norm(_quantity_to_value(model_out["metrics"]["j"], u.G / u.s), axis=-1)
    current_density[valid_mask] = np.trapz(j_norm, x=radius_cm, axis=0)
    return br, current_density


def main():
    parser = argparse.ArgumentParser(description="Compare AIA maps with a spherical NF2 extrapolation.")
    parser.add_argument("--spherical_nf2", type=str, required=True)
    parser.add_argument("--aia_193", type=str, required=True)
    parser.add_argument("--aia_304", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--hpc_x", type=float, nargs=2, default=None, metavar=("X_MIN", "X_MAX"),
                        help="Helioprojective x limits in arcsec.")
    parser.add_argument("--hpc_y", type=float, nargs=2, default=None, metavar=("Y_MIN", "Y_MAX"),
                        help="Helioprojective y limits in arcsec.")
    parser.add_argument("--radius_range", type=float, nargs=2, default=DEFAULT_RADIUS_RANGE,
                        metavar=("R_MIN", "R_MAX"), help="Radius range in solar radii.")
    parser.add_argument("--sampling", type=int, default=DEFAULT_SAMPLING,
                        help="Number of radial samples.")
    parser.add_argument("--batch_size", type=int, default=2 ** 14)
    parser.add_argument("--br_vmin", type=float, default=-1000)
    parser.add_argument("--br_vmax", type=float, default=1000)
    parser.add_argument("--aia_193_vmin", type=float, default=DEFAULT_AIA_193_VMIN)
    parser.add_argument("--aia_193_vmax", type=float, default=DEFAULT_AIA_193_VMAX)
    parser.add_argument("--aia_304_vmin", type=float, default=DEFAULT_AIA_304_VMIN)
    parser.add_argument("--aia_304_vmax", type=float, default=DEFAULT_AIA_304_VMAX)
    args = parser.parse_args()

    aia_193_map = _load_aia(args.aia_193, args.hpc_x, args.hpc_y)
    aia_304_map = _load_aia(args.aia_304, args.hpc_x, args.hpc_y)
    extent = _extent_from_hpc(aia_304_map)

    model = SphericalOutput(args.spherical_nf2)
    br, current_density = _load_spherical_projection(
        model,
        aia_304_map,
        args.radius_range,
        args.sampling,
        args.batch_size,
    )

    aia_193_norm = LogNorm(vmin=args.aia_193_vmin, vmax=args.aia_193_vmax)
    aia_304_norm = LogNorm(vmin=args.aia_304_vmin, vmax=args.aia_304_vmax)
    current_norm = _log_norm_from_positive(current_density)
    br_norm = Normalize(vmin=args.br_vmin, vmax=args.br_vmax)

    fig, axs = plt.subplots(1, 4, figsize=(13.5, 4.2), sharex=True, sharey=True)
    _plot_panel(fig, axs[0], aia_193_map.data, extent, _aia_colormap("193"),
                norm=aia_193_norm, label=r"AIA 193 $\AA$ [DN/s]")
    _plot_panel(fig, axs[1], aia_304_map.data, extent, _aia_colormap("304"),
                norm=aia_304_norm, label=r"AIA 304 $\AA$ [DN/s]")
    _plot_panel(fig, axs[2], br, extent, "gray", norm=br_norm, label=r"$B_r$ [G]")
    _plot_panel(fig, axs[3], current_density, extent, "inferno",
                norm=current_norm, label=r"$\int |J|\,dr$ [G cm s$^{-1}$]")

    for ax in axs:
        ax.set_xlabel("Helioprojective X [arcsec]")
    axs[0].set_ylabel("Helioprojective Y [arcsec]")
    fig.tight_layout()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_path, dpi=300, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
