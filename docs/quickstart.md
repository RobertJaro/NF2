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

This is the safest first run because NF2 generates the benchmark boundary internally. A successful run writes:

```text
runs/benchmark/case1/last.ckpt
runs/benchmark/case1/extrapolation_result.nf2
runs/benchmark/case1/work/
```

The analytical example is intentionally small, but it still trains a neural field. Runtime depends strongly on GPU availability; on a CUDA GPU it should feel like a smoke test, while CPU-only execution can be much slower. If the command reaches the end and writes `extrapolation_result.nf2`, the installation, configuration loading, training loop, checkpoint writing, and result loader path are all working.

Run a single observational extrapolation from a YAML configuration:

```bash
nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml \
  --run_path ./runs/ar377 \
  --work_path /scratch/ar377 \
  --Br /data/Br.fits \
  --Bt /data/Bt.fits \
  --Bp /data/Bp.fits
```

The observational command above is a template. Replace `/data/Br.fits`, `/data/Bt.fits`, and `/data/Bp.fits` with one matching vector magnetogram. For SHARP CEA data, `Br`, `Bt`, and `Bp` are the radial, theta, and phi components downloaded from JSOC. For generic Cartesian FITS arrays, use the `minimal_fits.yaml` example instead of `sharp_cea.yaml` unless your files include the SHARP/SunPy map metadata expected by the `sharp` loader.

Run a time series:

```bash
nf2-extrapolate-series --config examples/configs/cartesian/sharp_cea_series.yaml
```

Series runs need one completed single extrapolation before the series starts. First run the initial time step with `nf2-extrapolate`, then pass that run's `last.ckpt` as `--meta_path` and pass glob patterns such as `--Br_pattern "./data/*.Br.fits"`. Each component pattern must expand to the same number of files, and sorted filenames should pair the same time step across `Br`, `Bt`, `Bp`, and optional error maps. See [Series runs](series.md) for complete commands.

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
