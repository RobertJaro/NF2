# Notebook Workflows

Configurable Jupyter notebooks are provided for the main release workflows:

- `examples/notebooks/sharp_cea_cartesian.ipynb`
- `examples/notebooks/cartesian_series.ipynb`
- `examples/notebooks/spherical_hmi.ipynb`
- `examples/notebooks/benchmark_analytical.ipynb`

Each notebook exposes editable fields near the top for paths, active-region identifiers, time ranges, cadence, and run controls. The notebooks then walk through the full workflow: data download or file selection, Python API extrapolation, export, metrics, and visualization.

The visualization cells include current maps, free-energy maps, and boundary comparisons. Training cells are disabled by default through `RUN_TRAINING = False`; set that flag to `True` once the configuration and input files are ready.
