# Spherical Single Runs

Spherical runs combine full-disk HMI vector data with synoptic maps and train in a spherical coordinate volume. The example config is:

```text
examples/configs/spherical/hmi_full_disk.yaml
```

## 1. Download Full-Disk HMI Data

This command downloads a full-disk vector field and converts it to spherical Br/Bt/Bp components through JSOC's `HmiB2ptr` processing step.

```bash
nf2-download \
  --source hmi_full_disk \
  --download_dir "./data/hmi_spherical/full_disk" \
  --email "you@example.org" \
  --t_start 2011-02-15T00:00:00 \
  --series B_720s
```

## 2. Download Carrington Synoptic Maps

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --t_start "2011-02-15T00:00:00" \
  --series b_synoptic \
  --segments Br,Bt,Bp
```

You can also pass `--carrington_rotation` directly. Use `--synoptic_product mr_polfil` to download files such as `hmi.synoptic_mr_polfil_720s.2173.Mr_polfil.fits`.

## 3. Run The Spherical Config

Fill the placeholders with the downloaded files. The full-disk error placeholders can point to dedicated uncertainty maps; if those are not available, use the same field maps for a quick example and prepare proper uncertainties for production runs.

```bash
nf2-extrapolate \
  --config "examples/configs/spherical/hmi_full_disk.yaml" \
  --run_path "./runs/spherical_hmi" \
  --work_path "./runs/spherical_hmi/work" \
  --wandb_project "nf2" \
  --run_name "Spherical HMI" \
  --full_disk_Br "./data/hmi_spherical/full_disk/*Br.fits" \
  --full_disk_Bt "./data/hmi_spherical/full_disk/*Bt.fits" \
  --full_disk_Bp "./data/hmi_spherical/full_disk/*Bp.fits" \
  --full_disk_Br_err "./data/hmi_spherical/full_disk/*Br.fits" \
  --full_disk_Bt_err "./data/hmi_spherical/full_disk/*Bt.fits" \
  --full_disk_Bp_err "./data/hmi_spherical/full_disk/*Bp.fits" \
  --synoptic_Br "./data/hmi_spherical/synoptic/*Br.fits" \
  --synoptic_Bt "./data/hmi_spherical/synoptic/*Bt.fits" \
  --synoptic_Bp "./data/hmi_spherical/synoptic/*Bp.fits"
```

## 4. Memory Notes

Spherical runs are usually heavier than compact Cartesian cutouts. Reduce `batch_size`, `n_lat_lon_sample`, validation sampling, or radial/latitude/longitude ranges if you hit out-of-memory errors.
