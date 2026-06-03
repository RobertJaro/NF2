# Downloading Data

`nf2-download` provides a single command with source-specific download modes.

Most download modes use JSOC. Use the email address registered with JSOC, make sure DRMS/JSOC access works in the active Python environment, and expect large full-disk or time-series requests to take longer than a local file copy. After each download, check that every required component exists before starting NF2:

```bash
ls ./data/sharp_cea_377/*.Br.fits
ls ./data/sharp_cea_377/*.Bt.fits
ls ./data/sharp_cea_377/*.Bp.fits
```

For a single extrapolation, each component should resolve to one matching time step. For a series, each component pattern must resolve to the same number of files, and the sorted filenames should be chronological.

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
To inspect the mapping directly, use `nf2-noaa-to-sharp`:

```bash
nf2-noaa-to-sharp \
  --time 2011-02-15T00:00:00 \
  --email "you@example.org" \
  --noaa_nums 11158
```

SHARP CEA runs usually need `Br`, `Bt`, and `Bp`. Error maps `Br_err`, `Bt_err`, and `Bp_err` are optional in some configs but recommended when the example config exposes those placeholders. The filenames written by JSOC include the SHARP/HARP number, timestamp, and segment name, for example `hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits`.

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

Use the converted Br/Bt/Bp products for the spherical example configs. If full-disk uncertainty maps are available, pass them as `--full_disk_Br_err`, `--full_disk_Bt_err`, and `--full_disk_Bp_err`; for series runs use `--full_disk_Br_err_pattern`, `--full_disk_Bt_err_pattern`, and `--full_disk_Bp_err_pattern`, for example `./data/hmi_spherical/full_disk/*.Br_err.fits`. If you disable conversion, the native JSOC segments are useful for custom preprocessing but are not drop-in replacements for the `map` boundary examples.

## HMI Synoptic

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --t_start "2025-01-01T00:00:00" \
  --series b_synoptic \
  --segments Br,Bt,Bp
```

Use `--carrington_rotation` to select the rotation explicitly, or `--t_start` to infer it from a time. Use `--carrington_rotation_end` or `--t_end` for an inclusive range of rotations.

Download the polar-filled radial-field product with:

```bash
nf2-download \
  --source hmi_synoptic \
  --download_dir "./data/hmi_spherical/synoptic" \
  --email "you@example.org" \
  --carrington_rotation 2173 \
  --synoptic_product mr_polfil
```

Synoptic maps are usually paired with full-disk maps in spherical runs. Choose the Carrington rotation that covers the full-disk observation time, or let `--t_start` infer it. When mixing full-disk and synoptic constraints, keep the component naming consistent: NF2 expects spherical `Br`, `Bt`, and `Bp` maps unless your local config says otherwise.
