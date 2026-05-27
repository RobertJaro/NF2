# Quickstart

Install NF2 and activate the environment:

```bash
conda env create -f environment.yml
conda activate nf2
```

Run a single extrapolation from a YAML configuration:

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
nf2-extrapolate-series --config path/to/series.yaml
```

Load a result in Python:

```python
import nf2

model = nf2.load("path/to/extrapolation_result.nf2")
cube = model.load_cube(Mm_per_pixel=0.72, metrics=["j"])
```

Export a result:

```bash
nf2-export path/to/extrapolation_result.nf2 --format vtk --Mm_per_pixel 0.72 --metrics j alpha
```

Print standard NLFF quality metrics:

```bash
nf2-metrics path/to/extrapolation_result.nf2 --Mm_per_pixel 0.72 --height_range 0 80
```

Download example HMI/SHARP input data:

```bash
nf2-download --source hmi_sharp --download_dir ./data/sharp --email you@example.org --sharp_num 377 --t_start 2011-02-15T00:00:00
nf2-extrapolate --config examples/configs/cartesian/sharp_cea.yaml --run_path ./runs/ar377 --Br ./data/Br.fits --Bt ./data/Bt.fits --Bp ./data/Bp.fits
```

Use the generated reference pages for complete command and configuration options.
For interactive walkthroughs, open the configurable notebooks in `examples/notebooks/`.
