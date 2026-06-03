# Spherical Series Runs

NF2 can run spherical sequences when the spherical boundary file entries are glob patterns or file lists. Use `nf2/spherical/hmi_full_disk.yaml` for the initial extrapolation and `nf2/spherical/hmi_full_disk_series.yaml` for the sequence.

As with Cartesian series, a spherical series needs the completed first extrapolation's `last.ckpt` as `meta_path`.

## 1. Run The First Spherical Extrapolation

Use the single-run guide in [spherical.md](spherical.md) for the first time step:

```bash
nf2-extrapolate \
  --config "nf2/spherical/hmi_full_disk.yaml" \
  --run_path "./runs/spherical_initial" \
  --work_path "./runs/spherical_initial/work" \
  --wandb_project "nf2" \
  --run_name "Spherical HMI initial" \
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

The `--full_disk_Br_err`, `--full_disk_Bt_err`, and `--full_disk_Bp_err` arguments are optional. If you omit them, NF2 skips the full-disk error maps.

## 2. Use The Series Config

Use the series config directly and fill its `<<...>>` placeholders from the command line. Every glob must match either one shared file or the same number of time steps as the other series components.

Within a spherical series config, `[[dataset.path.to.value]]` references are resolved after dataset files are expanded. For example, `[[full_disk.files.Br]]` points to the current time step's full-disk `Br` file and is not a command-line override.

The series template validates and logs every 10th dataset by default while still saving one `.nf2` file per dataset.

## 3. Run The Series

```bash
nf2-extrapolate-series \
  --config "nf2/spherical/hmi_full_disk_series.yaml" \
  --run_path "./runs/spherical_series" \
  --work_path "./runs/spherical_series/work" \
  --meta_path "./runs/spherical_initial/last.ckpt" \
  --wandb_project "nf2" \
  --run_name "Spherical HMI series" \
  --full_disk_Br_pattern "./data/hmi_spherical/full_disk/*.Br.fits" \
  --full_disk_Bt_pattern "./data/hmi_spherical/full_disk/*.Bt.fits" \
  --full_disk_Bp_pattern "./data/hmi_spherical/full_disk/*.Bp.fits" \
  --full_disk_Br_err_pattern "./data/hmi_spherical/full_disk/*.Br_err.fits" \
  --full_disk_Bt_err_pattern "./data/hmi_spherical/full_disk/*.Bt_err.fits" \
  --full_disk_Bp_err_pattern "./data/hmi_spherical/full_disk/*.Bp_err.fits" \
  --synoptic_Br_pattern "./data/hmi_spherical/synoptic/*.Br.fits" \
  --synoptic_Bt_pattern "./data/hmi_spherical/synoptic/*.Bt.fits" \
  --synoptic_Bp_pattern "./data/hmi_spherical/synoptic/*.Bp.fits" \
  --fits_reference_Br "./data/hmi_spherical/reference/20240510_000000.Br.fits"
```

The `--full_disk_Br_err_pattern`, `--full_disk_Bt_err_pattern`, and `--full_disk_Bp_err_pattern` arguments are optional. If you omit them, NF2 runs the spherical series without full-disk error maps.

## 4. Export The Series

```bash
nf2-export "./runs/spherical_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/spherical_series/exports" \
  --metrics j alpha free_energy_fft \
  --overwrite
```
