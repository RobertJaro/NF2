# Series Runs

Series runs reuse the same public YAML schema, but placeholders usually point to glob patterns instead of single files.

A series run requires an initial single extrapolation first. Run the first time step with `nf2-extrapolate`, then pass that first run's `extrapolation_result.nf2` as `--meta_path` when starting the full series. The meta path is the start point for the series and lets each subsequent time step continue from a trained NF2 state instead of starting cold.

## Step 1: Run The Initial Extrapolation

Use the matching single-run config for the first frame in the sequence:

```bash
nf2-extrapolate \
  --config "examples/configs/cartesian/multi_height.yaml" \
  --run_path "./runs/multi_height_initial" \
  --work_path "./runs/multi_height_initial/work" \
  --photosphere_B_los "./data/photosphere/20240101_000000_B_los.fits" \
  --photosphere_B_trv "./data/photosphere/20240101_000000_B_trv.fits" \
  --photosphere_B_azi "./data/photosphere/20240101_000000_B_azi.fits"
```

Wait until this run writes `./runs/multi_height_initial/extrapolation_result.nf2`.

## Step 2: Start The Full Series

```bash
nf2-extrapolate-series \
  --config "examples/configs/cartesian/multi_height_series.yaml" \
  --run_path "./runs/multi_height_series" \
  --work_path "./runs/multi_height_series/work" \
  --meta_path "./runs/multi_height_initial/extrapolation_result.nf2" \
  --photosphere_B_los_pattern "./data/photosphere/*B_los.fits" \
  --photosphere_B_trv_pattern "./data/photosphere/*B_trv.fits" \
  --photosphere_B_azi_pattern "./data/photosphere/*B_azi.fits"
```

The `--meta_path` value must point to the completed first extrapolation. Keep that first run available until the series has started successfully.

Each component pattern in a boundary must expand to the same number of files. NF2 pairs files by sorted order, so use filenames that sort chronologically and consistently across components.

Multi-height extrapolations can be used as series runs when all boundary heights provide matching file sequences. First create the single-run meta path from one complete multi-height extrapolation, then start the full multi-height series with glob patterns for every photospheric and chromospheric component.
