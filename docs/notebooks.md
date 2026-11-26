# Notebook Examples

Configurable Jupyter notebooks are provided for the main release examples:

- [SHARP CEA Cartesian](../examples/notebooks/sharp_cea_cartesian.ipynb)
- [Cartesian series](../examples/notebooks/cartesian_series.ipynb)
- [Spherical HMI](../examples/notebooks/spherical_hmi.ipynb)
- [Analytical benchmark](../examples/notebooks/benchmark_analytical.ipynb)
- [Field-line tracing from an NF2 file](../examples/notebooks/field_line_tracing.ipynb)

Colab-ready notebooks are available for the most common remote workflows:

- [SHARP CEA tutorial](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/colab_sharp_cea.ipynb)
- [SHARP CEA Cartesian](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/sharp_cea_cartesian.ipynb)
- [HMI spherical](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/examples/notebooks/spherical_hmi.ipynb)

Each notebook exposes editable fields near the top for paths, active-region identifiers, time ranges, cadence, and run controls. The notebooks then walk through the full analysis path: data download or file selection, Python API extrapolation, export, metrics, and visualization.

The visualization cells include current maps, free-energy maps, and boundary comparisons. The SHARP and Cartesian series notebooks enable download and training by default. To inspect a config first, disable its run flags in the settings cell. GPU runtime is recommended for training; CPU runtime is usually only practical for configuration checks and already-trained outputs.
