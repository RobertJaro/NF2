# Export

Use the unified export command:

```bash
nf2-export --checkpoint /path/to/result.nf2 --format vtk
```

Supported formats:

- `vtk`
- `hdf5`
- `fits`
- `npz`
- `binary`

Series export:

```bash
nf2-export --series --checkpoint "/path/to/results/*.nf2" --format hdf5 --out_dir /path/to/exports
```

Legacy convenience commands still point at the shared exporter:

- `nf2-to-vtk`
- `nf2-to-hdf5`
- `nf2-to-fits`
- `nf2-to-npz`
- `nf2-to-binary`
