# Cartesian Single Runs

Cartesian runs train a local volume above one or more planar boundary maps. Use these examples for SHARP CEA data, generic plain FITS arrays, and multi-height observations.

## SHARP CEA Data

First download a SHARP CEA vector magnetogram from JSOC. Fill in your JSOC email and either a SHARP number or NOAA number.

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir "./data/sharp_cea_377" \
  --email "you@example.org" \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --t_end 2011-02-15T00:12:00 \
  --cadence 720s \
  --series sharp_cea_720s \
  --segments Br,Bp,Bt,Br_err,Bp_err,Bt_err
```

Then run the SHARP CEA config. The `sharp` dataset reads SHARP CEA maps with map metadata. Error-map arguments are optional; omit `--Br_err`, `--Bt_err`, and `--Bp_err` if those files are not available.

```bash
nf2-extrapolate \
  --config "nf2/cartesian/sharp_cea.yaml" \
  --run_path "./runs/sharp_cea_377" \
  --work_path "./runs/sharp_cea_377/work" \
  --Br "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits" \
  --Bt "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits" \
  --Bp "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits" \
  --Br_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits" \
  --Bt_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits" \
  --Bp_err "./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
```

## Plain FITS Arrays

Use `minimal_fits.yaml` for plain Br/Bt/Bp FITS data arrays without SunPy map metadata. This config uses `type: fits` with `load_map: false`.

```bash
nf2-extrapolate \
  --config "nf2/cartesian/minimal_fits.yaml" \
  --run_path "./runs/cartesian_minimal" \
  --Br "./data/plain_fits/Br.fits" \
  --Bt "./data/plain_fits/Bt.fits" \
  --Bp "./data/plain_fits/Bp.fits"
```

Use this path when your FITS files only contain array data and do not need coordinate information from FITS/WCS headers.

Use `--z_range z_min z_max` with any Cartesian extrapolation command to adjust the height of the extrapolated volume to the size of the horizontal domain. For example, `--z_range 0 150` raises the top boundary to 150 Mm; the bundled Cartesian default is `[0, 100]`.

## Multi-Height Cartesian Data

Multi-height examples assume you already prepared matching photospheric and chromospheric FITS files. No generic download command is provided because the source instruments and preprocessing are project-specific.

```bash
nf2-extrapolate \
  --config "nf2/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/multi_height/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/multi_height/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/multi_height/photosphere/20240101_000000_B_azi.fits" \
  --chromosphere_B_los "./data/multi_height/chromosphere/20240101_000000_B_los.fits" \
  --chromosphere_B_trv "./data/multi_height/chromosphere/20240101_000000_B_trv.fits" \
  --chromosphere_B_azi "./data/multi_height/chromosphere/20240101_000000_B_azi.fits"
```

Keep this first multi-height run if you plan to start a series. Its `last.ckpt` becomes the `meta_path` for the series run.

## Common Adjustments

- Reduce `data.sampler.batch_size`, boundary `batch_size`, or validation `batch_size` if training runs out of GPU memory.
- Use `WANDB_MODE=offline` or `WANDB_MODE=disabled` if you do not want online W&B logging.
- Run `nf2-metrics` after training to check boundary quality, divergence, current alignment, and magnetic energy.
