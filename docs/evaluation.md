# Evaluation

Use the evaluation pages after training has produced an `extrapolation_result.nf2`.

- [Exporting](exporting.md) writes field cubes and derived quantities to VTK, NPZ, HDF5, or FITS.
- [Quality metrics](metrics.md) computes divergence, force-free, current-alignment, and magnetic-energy diagnostics.

Choose the output based on the next tool in your workflow:

| Goal | Use | Notes |
| --- | --- | --- |
| Inspect field lines or volumes in ParaView | `nf2-export --format vtk` | Best for visual quality checks and presentations. |
| Continue analysis in Python | `nf2-export --format npz` | Keeps arrays easy to load with NumPy. |
| Share structured data with other tools | `nf2-export --format hdf5` | Useful for larger runs and series. |
| Compare physical quality across runs | `nf2-metrics` | Reports divergence, force-free alignment, and energy diagnostics. |
| Inspect learned multi-height surfaces | `nf2-export --format height` | Requires a checkpoint with height mappings. |

```{toctree}
:maxdepth: 2
:caption: Evaluation

exporting
metrics
```
