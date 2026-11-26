# Evaluating checkpoint metrics

Run `nf2 metrics` on a trained checkpoint to print a table of standard quality
and energy diagnostics. To see every metric and its unit without loading a checkpoint, run
`nf2 metrics --list-metrics`.

```bash
nf2 metrics model.nf2
nf2 metrics model.nf2 --metrics rms_divB theta_J E_free --output quality.csv
nf2 metrics model.nf2 --metrics standard flux helicity --Mm_per_pixel 0.72 --height_range 0 100 --output report.json
nf2 metrics spherical.nf2 --metrics energies strength --spherical_sampling 32 64 128 --output energies.txt
```

An explicit list replaces the default selection. Names and groups can be
combined in any order. Include `standard` to add optional quantities to the
usual report. Each quantity appears once. In Python,
`nf2.metrics.compute_metrics(path, metrics=["quality", "E_tot"])` returns
a flat dictionary of scalar values and the checkpoint geometry.

| Group | Quantities |
| --- | --- |
| `standard` | `quality` and the energies available for this geometry. |
| `quality` | Mean absolute and RMS divergence, both absolute and normalized by local field strength. Current-weighted alignment angle and sine (`theta_J`, `sigma_J`, alias `CWsin`). Mean and RMS force-free residual. |
| `energies` | Volume-integrated `E_tot`. Cartesian reports also include `E_pot`, `E_free`, and `E_free_over_E_tot`. The FFT reference uses the vertical field at the lowest sampled layer. |
| `strength` | `mean_B` and `max_B`, in gauss. |
| `flux` | Cartesian signed outward flux, unsigned flux over all six faces, unsigned flux through the lower face, and absolute net flux divided by unsigned flux. |
| `helicity` | Cartesian relative, current-carrying, and mutual helicities, each reconstructed from the bottom and top faces. Includes net flux, flux imbalance, the uniform normal-field correction, and RMS grid divergence. |

## Sampling and reports

Cartesian ranges (`--x_range`, `--y_range`, `--height_range`) and
`--Mm_per_pixel` use megameters. For spherical checkpoints specify
`--radius_range` in solar radii, angular ranges in degrees, and
`--spherical_sampling` in radius, latitude, longitude order. Use the same
bounds and resolution when comparing checkpoints. The batch size controls
how many coordinates the network evaluates per forward pass.

Quality metrics use the neural-field Jacobian. Energy and strength evaluation
only require the field. Flux alone evaluates the six faces. Helicity uses
a grid with nodes on both ends of each axis and reports its actual spacings.
The standard Cartesian sampling convention places nodes starting at the lower
bounds and uses the requested spacing for voxel energies. A combined report
records both grids. For helicity, `rms_div_B_G_per_Mm` measures finite-difference
divergence on the closed box. The standard `rms_divB` uses automatic differentiation.

The terminal and `.txt` outputs give a human-readable table. CSV contains one
row per scalar with its numeric value, unit, checkpoint, geometry and sampling
metadata. JSON contains the same scalars and a shared grid description.
Integration direction is included in directional metric names, e.g.
`H_V_bottom` and `H_V_top`. Undefined values appear as `n/a` in text, empty
CSV values, and JSON `null`.

The [generated reference](generated/export_metrics_reference.md) lists every scalar.
Export metrics such as `magnetic_helicity` (local A·B) produce spatial data.
The `helicity` group integrates relative magnetic helicity over the requested box.

## Relative magnetic helicity in a Cartesian box

Evaluate a Cartesian checkpoint with either field representation. A coarser grid
is useful for a first run. Halving the spacing increases the number of nodes
and approximate solver memory by a factor of eight for a large 3D box.

```bash
python -m nf2.evaluation.helicity extrapolation_result.nf2 --spacing-Mm 1.44 > helicity.json
```

The evaluator samples the full box, including all six faces. Changing the
spacing preserves the physical volume. The JSON gives the actual spacing,
box limits, and two estimates of total relative helicity (`H_V_Mx2`),
current-carrying helicity (`H_J_Mx2`), and their mutual term (`H_PJ_Mx2`),
all in Mx². For an existing array, call `relative_helicity(b_gauss, spacing_Mm)`
in `nf2.evaluation.helicity` with axes ordered `(x, y, z, component)` in a
right-handed frame.

The reference field is the gradient of a scalar potential obtained by solving
the Neumann problem using normal field data on all six faces. A nonzero net
flux makes those data incompatible. The solver removes the reported constant
outward-field offset from the reference boundary data. The input field is used
as supplied in the helicity integrals. The two estimates reconstruct DeVore
vector potentials starting at opposite faces and apply the Finn–Antonsen volume
integral. For a resolved solenoidal field the two estimates should converge.
[Valori et al. (2012)](https://doi.org/10.1007/s11207-012-9951-6)
and [Valori et al. (2016)](https://doi.org/10.1007/s11214-016-0299-3).

For a sign comparison, inspect the signed values from both faces and repeat
at finer spacing. Also inspect the absolute net flux divided by total unsigned
face flux (`flux_imbalance_fraction`) and the RMS divergence on the sampled
grid. A substantial flux correction or disagreement between the two estimates
makes the sign questionable. Agreement alone cannot establish solenoidal
quality. An independent finite-volume result on the same field and box gives
a stronger check.
