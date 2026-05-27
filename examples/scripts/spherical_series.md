# Spherical Series Runs

NF2 can run spherical sequences when the spherical boundary file entries are glob patterns or file lists. The repository currently ships a single-run spherical template, `examples/configs/spherical/full_disk_synoptic.yaml`; adapt it for a series by using glob patterns that match one full-disk and synoptic set per time step.

As with Cartesian series, a spherical series needs a completed first extrapolation as `meta_path`.

## 1. Run The First Spherical Extrapolation

Use the single-run guide in [spherical.md](spherical.md) for the first time step:

```bash
nf2-extrapolate \
  --config "examples/configs/spherical/full_disk_synoptic.yaml" \
  --run_path "./runs/spherical_initial" \
  --work_path "./runs/spherical_initial/work" \
  --full_disk_Br "./data/hmi_spherical/full_disk/20240510_000000.Br.fits" \
  --full_disk_Bt "./data/hmi_spherical/full_disk/20240510_000000.Bt.fits" \
  --full_disk_Bp "./data/hmi_spherical/full_disk/20240510_000000.Bp.fits" \
  --full_disk_Br_err "./data/hmi_spherical/full_disk/20240510_000000.Br_err.fits" \
  --full_disk_Bt_err "./data/hmi_spherical/full_disk/20240510_000000.Bt_err.fits" \
  --full_disk_Bp_err "./data/hmi_spherical/full_disk/20240510_000000.Bp_err.fits" \
  --synoptic_Br "./data/hmi_spherical/synoptic/2283.Br.fits" \
  --synoptic_Bt "./data/hmi_spherical/synoptic/2283.Bt.fits" \
  --synoptic_Bp "./data/hmi_spherical/synoptic/2283.Bp.fits"
```

## 2. Create A Series Config

Copy the spherical config and replace each single file placeholder with a glob placeholder, for example:

```yaml
path: ./runs/spherical_series
work_path: ./runs/spherical_series/work
meta_path: ./runs/spherical_initial/extrapolation_result.nf2
data:
  geometry: spherical
  boundaries:
    - id: full_disk
      type: map
      files:
        Br: ./data/hmi_spherical/full_disk/*.Br.fits
        Bt: ./data/hmi_spherical/full_disk/*.Bt.fits
        Bp: ./data/hmi_spherical/full_disk/*.Bp.fits
      errors:
        Br_err: ./data/hmi_spherical/full_disk/*.Br_err.fits
        Bt_err: ./data/hmi_spherical/full_disk/*.Bt_err.fits
        Bp_err: ./data/hmi_spherical/full_disk/*.Bp_err.fits
    - id: synoptic
      type: map
      files:
        Br: ./data/hmi_spherical/synoptic/*.Br.fits
        Bt: ./data/hmi_spherical/synoptic/*.Bt.fits
        Bp: ./data/hmi_spherical/synoptic/*.Bp.fits
```

Every glob must match either one shared file or the same number of time steps as the other series components.

## 3. Run The Series

```bash
nf2-extrapolate-series \
  --config "./my_spherical_series.yaml"
```

## 4. Export The Series

```bash
nf2-export "./runs/spherical_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/spherical_series/exports" \
  --metrics j alpha free_energy \
  --overwrite
```
