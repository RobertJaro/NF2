# Series Runs

Series runs reuse the same public YAML schema, but placeholders usually point to glob patterns instead of single files.

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

Each component pattern in a boundary must expand to the same number of files. NF2 pairs files by sorted order, so use filenames that sort chronologically and consistently across components.

Multi-height extrapolations can be used as series runs when all boundary heights provide matching file sequences.
