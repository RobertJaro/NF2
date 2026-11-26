# Export And Metrics Reference

## Export Metrics

| Metric | Output Key(s) | Description |
| --- | --- | --- |
| j | j | Current-density vector. |
| alpha | alpha | Force-free alpha, computed as (J . B) / \|B\|^2. |
| b_nabla_bz | b_nabla_bz | Vertical magnetic tension-related derivative. |
| energy | energy | Magnetic energy density. |
| energy_gradient | energy_gradient | Cartesian vertical magnetic-energy gradient. |
| spherical_energy_gradient | spherical_energy_gradient | Spherical radial magnetic-energy gradient. |
| free_energy | free_energy | Cartesian free magnetic energy density using the default potential-field method. |
| free_energy_fft | free_energy_fft | Free magnetic energy density using the Cartesian FFT potential field. |
| free_energy_direct | free_energy_direct | Free magnetic energy density using the direct potential-field method. |
| magnetic_helicity | magnetic_helicity | Magnetic helicity diagnostic. |
| los_trv_azi | los_trv_azi | LOS/transverse/azimuth field components. |
| squashing_factor | squashing_factor, log10_q, q_valid, q_condition_number | Boundary-to-boundary squashing factor from batched NF2 field-line tracing. |
| twist_number | twist_number | Field-line twist number Tw = integral(alpha dl) / (4 pi). |
| fieldline_length | fieldline_length | Total boundary-to-boundary field-line length. |
| integrated_current_density | integrated_current_density | Vector current density integrated over field-line length. |
| fieldline_geometry | open, closed, open_polarity, footpoint_separation, apex_height, apex_radius | Open/closed connectivity, apex, open polarity, and footpoint separation. |

## Quality Metrics

| Metric | Group | Unit | Geometry | Description |
| --- | --- | --- | --- | --- |
| mean_abs_divB | quality | G / Mm | Cartesian and spherical | Mean absolute divergence. |
| rms_divB | quality | G / Mm | Cartesian and spherical | Root-mean-square divergence. |
| mean_abs_divB_over_B | quality | 1 / Mm | Cartesian and spherical | Mean absolute divergence divided by local field strength. |
| rms_divB_over_B | quality | 1 / Mm | Cartesian and spherical | RMS divergence divided locally by field strength. |
| theta_J | quality | deg | Cartesian and spherical | Arcsine of the current-weighted mean sine of the alignment angle. |
| sigma_J | quality | 1 | Cartesian and spherical | Current-weighted mean sine of the alignment angle (also CWsin). |
| mean_force_free_residual | quality | G / Mm | Cartesian and spherical | Mean magnitude of curl(B) perpendicular to B. |
| rms_force_free_residual | quality | G / Mm | Cartesian and spherical | RMS magnitude of curl(B) perpendicular to B. |
| E_tot | energies | erg | Cartesian and spherical | Total magnetic energy. |
| E_pot | energies | erg | Cartesian | Potential energy from the lower-boundary FFT reference. |
| E_free | energies | erg | Cartesian | Total energy minus FFT potential energy. |
| E_free_over_E_tot | energies | 1 | Cartesian | Fraction of total energy in excess of FFT potential energy. |
| mean_B | strength | G | Cartesian and spherical | Mean field strength. |
| max_B | strength | G | Cartesian and spherical | Maximum field strength. |
| net_flux_Mx | flux | Mx | Cartesian | Signed outward flux through all six faces. |
| unsigned_flux_Mx | flux | Mx | Cartesian | Unsigned flux through all six faces. |
| bottom_unsigned_flux_Mx | flux | Mx | Cartesian | Unsigned flux through the lower face. |
| flux_imbalance_fraction | flux | 1 | Cartesian | Absolute net flux divided by unsigned flux (zero for zero flux). |
| outward_field_offset_removed_G | helicity | G | Cartesian | Uniform outward offset removed from reference data. |
| rms_div_B_G_per_Mm | helicity | G / Mm | Cartesian | RMS finite-difference divergence on the closed box. |
| H_V_bottom | helicity | Mx² | Cartesian | Total relative helicity integrated from the bottom. |
| H_V_top | helicity | Mx² | Cartesian | Total relative helicity integrated from the top. |
| H_J_bottom | helicity | Mx² | Cartesian | Current-carrying helicity integrated from the bottom. |
| H_J_top | helicity | Mx² | Cartesian | Current-carrying helicity integrated from the top. |
| H_PJ_bottom | helicity | Mx² | Cartesian | Mutual helicity integrated from the bottom. |
| H_PJ_top | helicity | Mx² | Cartesian | Mutual helicity integrated from the top. |
