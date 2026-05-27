# Cartesian Series Runs

Series runs train a sequence of extrapolations. They require an initial single extrapolation first. That first run provides the `meta_path` start point, so the series continues from a trained NF2 state instead of starting each time step from random weights.

## 1. Prepare The Initial Run

For multi-height data, first run one complete single extrapolation for the first time step:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/multi_height/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/multi_height/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/multi_height/photosphere/20240101_000000_B_azi.fits" \
  --chromosphere_B_los "./data/multi_height/chromosphere/20240101_000000_B_los.fits" \
  --chromosphere_B_trv "./data/multi_height/chromosphere/20240101_000000_B_trv.fits" \
  --chromosphere_B_azi "./data/multi_height/chromosphere/20240101_000000_B_azi.fits"
```

The series command below uses:

```text
./runs/multi_height_initial/extrapolation_result.nf2
```

as its `meta_path`.

## 2. Check File Patterns

Every component pattern must match the same number of files. Filenames should sort in chronological order. Check this before starting a long run:

```bash
ls ./data/multi_height/photosphere/*B_los.fits | head
ls ./data/multi_height/chromosphere/*B_los.fits | head
```

## 3. Run The Series

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --work_path "./runs/multi_height_series/work" \
  --meta_path "./runs/multi_height_initial/extrapolation_result.nf2" \
  --photosphere_B_los_pattern "./data/multi_height/photosphere/*B_los.fits" \
  --photosphere_B_trv_pattern "./data/multi_height/photosphere/*B_trv.fits" \
  --photosphere_B_azi_pattern "./data/multi_height/photosphere/*B_azi.fits" \
  --chromosphere_B_los_pattern "./data/multi_height/chromosphere/*B_los.fits" \
  --chromosphere_B_trv_pattern "./data/multi_height/chromosphere/*B_trv.fits" \
  --chromosphere_B_azi_pattern "./data/multi_height/chromosphere/*B_azi.fits"
```

## 4. Export The Series

```bash
nf2-export "./runs/multi_height_series/*.nf2" \
  --format hdf5 \
  --out-dir "./runs/multi_height_series/exports" \
  --Mm_per_pixel 1.44 \
  --height_range 0 100 \
  --metrics j alpha free_energy \
  --overwrite
```

If a series step fails, inspect that time step's input files first. Mismatched glob counts, missing error maps, or non-chronological filenames are the most common setup issues.
