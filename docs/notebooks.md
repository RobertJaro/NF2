# Notebook Examples

Configurable Jupyter notebooks are provided for the main release examples:

- [SHARP CEA Cartesian](../examples/notebooks/sharp_cea_cartesian.ipynb)
- [Cartesian series](../examples/notebooks/cartesian_series.ipynb)
- [Spherical HMI](../examples/notebooks/spherical_hmi.ipynb)
- [Analytical benchmark](../examples/notebooks/benchmark_analytical.ipynb)

Colab-ready notebooks are available for the most common remote workflows:

- [SHARP CEA tutorial](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/colab_sharp_cea.ipynb)
- [SHARP CEA Cartesian](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/sharp_cea_cartesian.ipynb)
- [HMI spherical](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/spherical_hmi.ipynb)

Each notebook exposes editable fields near the top for paths, active-region identifiers, time ranges, cadence, and run controls. The notebooks then walk through the full analysis path: data download or file selection, Python API extrapolation, export, metrics, and visualization.

The visualization cells include current maps, free-energy maps, and boundary comparisons. Training cells are disabled by default through `RUN_TRAINING = False`; set that flag to `True` once the configuration and input files are ready. GPU runtime is recommended for training; CPU runtime is usually only practical for configuration checks and already-trained outputs.
