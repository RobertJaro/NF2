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
| squashing_factor | squashing_factor, twist | Squashing factor and twist diagnostics. |

## Quality Metrics

| Metric | Description |
| --- | --- |
| mean_abs_divB | Mean absolute divergence in G/Mm. |
| rms_divB | Root-mean-square divergence in G/Mm. |
| mean_abs_divB_over_B | Mean \|div B\| / \|B\| in 1/Mm. |
| rms_divB_over_B | Root-mean-square \|div B\| / \|B\| in 1/Mm. |
| theta_J | Current-weighted angle between J and B in degrees. |
| sigma_J | Current-weighted sine of the J-B angle. |
| CWsin | Alias for sigma_J used in NLFF comparisons. |
| mean_force_free_residual | Mean \|J x B\| / \|B\|. |
| rms_force_free_residual | Root-mean-square \|J x B\| / \|B\|. |
| E_tot | Total magnetic energy. |
| E_free | Free magnetic energy. Cartesian uses the FFT potential-field reference. |
| E_pot | Potential magnetic energy estimate E_tot - E_free. |
| E_free_over_E_tot | Free-energy fraction. |
| mean_B | Mean magnetic-field strength. |
| max_B | Maximum magnetic-field strength. |
