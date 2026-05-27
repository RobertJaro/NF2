# Downloading Data

`nf2-download` provides a single command with source-specific download modes.

## SHARP CEA

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

Use `--noaa_num` instead of `--sharp_num` when the NOAA active-region number should be resolved through JSOC metadata.

## HMI Full Disk

```bash
nf2-download \
  --source hmi_full_disk \
  --download_dir "./data/hmi_spherical/full_disk" \
  --email "you@example.org" \
  --t_start 2024-05-10T00:00:00 \
  --series B_720s
```

By default, full-disk vector data are converted to Br/Bt/Bp with JSOC `HmiB2ptr`. Use `--no_convert_ptr` to download the native segments.

## HMI Synoptic

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --carrington_rotation 2283 \
  --series b_synoptic \
  --segments Br,Bt,Bp
```

Use `--carrington_rotation_end` for an inclusive range of rotations.
