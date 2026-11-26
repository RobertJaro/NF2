"""Scalar diagnostics exposed by the metrics CLI and Python API."""
from typing import NamedTuple


class Metric(NamedTuple):
    group: str
    unit: str
    description: str
    cartesian_only: bool = False


METRICS = {
    "mean_abs_divB": Metric("quality", "G / Mm", "Mean absolute divergence."),
    "rms_divB": Metric("quality", "G / Mm", "Root-mean-square divergence."),
    "mean_abs_divB_over_B": Metric("quality", "1 / Mm", "Mean absolute divergence divided by local field strength."),
    "rms_divB_over_B": Metric("quality", "1 / Mm", "RMS divergence divided locally by field strength."),
    "theta_J": Metric("quality", "deg", "Arcsine of the current-weighted mean sine of the alignment angle."),
    "sigma_J": Metric("quality", "1", "Current-weighted mean sine of the alignment angle (also CWsin)."),
    "mean_force_free_residual": Metric("quality", "G / Mm", "Mean magnitude of curl(B) perpendicular to B."),
    "rms_force_free_residual": Metric("quality", "G / Mm", "RMS magnitude of curl(B) perpendicular to B."),
    "E_tot": Metric("energies", "erg", "Total magnetic energy."),
    "E_pot": Metric("energies", "erg", "Potential energy from the lower-boundary FFT reference.", True),
    "E_free": Metric("energies", "erg", "Total energy minus FFT potential energy.", True),
    "E_free_over_E_tot": Metric("energies", "1", "Fraction of total energy in excess of FFT potential energy.", True),
    "mean_B": Metric("strength", "G", "Mean field strength."),
    "max_B": Metric("strength", "G", "Maximum field strength."),
    "net_flux_Mx": Metric("flux", "Mx", "Signed outward flux through all six faces.", True),
    "unsigned_flux_Mx": Metric("flux", "Mx", "Unsigned flux through all six faces.", True),
    "bottom_unsigned_flux_Mx": Metric("flux", "Mx", "Unsigned flux through the lower face.", True),
    "flux_imbalance_fraction": Metric("flux", "1", "Absolute net flux divided by unsigned flux (zero for zero flux).", True),
    "outward_field_offset_removed_G": Metric("helicity", "G", "Uniform outward offset removed from reference data.", True),
    "rms_div_B_G_per_Mm": Metric("helicity", "G / Mm", "RMS finite-difference divergence on the closed box.", True),
}

for component, label in (("H_V", "Total relative"), ("H_J", "Current-carrying"), ("H_PJ", "Mutual")):
    for face in ("bottom", "top"):
        METRICS[f"{component}_{face}"] = Metric("helicity", "Mx²", f"{label} helicity integrated from the {face}.", True)

GROUPS = {group: [name for name, metric in METRICS.items() if metric.group == group]
          for group in ("quality", "energies", "strength", "flux", "helicity")}
GROUPS["helicity"] += ["net_flux_Mx", "flux_imbalance_fraction"]
GROUPS["standard"] = GROUPS["quality"] + GROUPS["energies"]


def select_metrics(names, geometry=None):
    """Expand groups in request order. Groups adapt to available energies."""
    if names is None:
        names = ["standard"]
    elif isinstance(names, str):
        names = names.split()
    if not names:
        raise ValueError("Select at least one metric or group.")
    selected = {}
    for token in names:
        token = "sigma_J" if token == "CWsin" else token
        if token not in GROUPS and token not in METRICS:
            raise ValueError(f"Unknown metric '{token}'. See --list-metrics.")
        for name in GROUPS.get(token, [token]):
            if geometry == "spherical" and METRICS[name].cartesian_only:
                if token in ("standard", "energies"):
                    continue
                raise ValueError(f"'{token}' requires a Cartesian checkpoint.")
            selected[name] = None
    return list(selected)
