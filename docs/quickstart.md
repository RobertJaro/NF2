# Quickstart

Install NF2 in a fresh Python environment:

```bash
pip install nf2
```

For a local source checkout, use the repository environment instead:

```bash
conda env create -f environment.yml
conda activate nf2
```

Run a compact analytical smoke test. This does not require JSOC access or input FITS files:

```bash
nf2-extrapolate \
  --config examples/configs/benchmark/analytical_case1.yaml \
  --run_path ./runs/benchmark/case1 \
  --work_path ./runs/benchmark/case1/work
```

Run a single observational extrapolation from a YAML configuration:

```bash
nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/ar377 \
  --work_path /scratch/ar377 \
  --Br /data/Br.fits \
  --Bt /data/Bt.fits \
  --Bp /data/Bp.fits
```

Run a time series:

```bash
nf2-extrapolate-series --config examples/configs/cartesian/sharp_cea_series.yaml
```

Load a result in Python:

```python
import nf2

model = nf2.load("./runs/benchmark/case1/extrapolation_result.nf2")
cube = model.load_cube(Mm_per_pixel=0.72, metrics=["j"])
```

Export a result:

```bash
nf2-export ./runs/benchmark/case1/extrapolation_result.nf2 --format vtk --Mm_per_pixel 0.72 --metrics j alpha
```

Print standard NLFF quality metrics:

```bash
nf2-metrics ./runs/benchmark/case1/extrapolation_result.nf2 --Mm_per_pixel 0.05 --height_range 0 2
```

Download an example HMI SHARP CEA magnetogram from JSOC:

```bash
nf2-download \
  --source hmi_sharp \
  --download_dir ./data/sharp_cea_377 \
  --email you@example.org \
  --sharp_num 377 \
  --t_start 2011-02-15T00:00:00 \
  --series sharp_cea_720s \
  --segments Br,Bt,Bp,Br_err,Bt_err,Bp_err
```

Replace `you@example.org` with the email address registered with JSOC.
Use `--noaa_num` instead of `--sharp_num` if you want NF2 to resolve the SHARP/HARP number from a NOAA active-region number.
The command above downloads one 720 s SHARP CEA time step and writes JSOC-style filenames such as `hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits`.
Add `--t_end` and `--cadence 720s` when downloading a SHARP time sequence.

Run the extrapolation with the downloaded files:

```bash
nf2-extrapolate \
  --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/ar377 \
  --Br ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits \
  --Bt ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits \
  --Bp ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits \
  --Br_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits \
  --Bt_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits \
  --Bp_err ./data/sharp_cea_377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits
```

Use the generated reference pages for complete command and configuration options.
For interactive walkthroughs, open the configurable notebooks in `examples/notebooks/`.
