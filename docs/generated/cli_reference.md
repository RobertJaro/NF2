# CLI Reference

## `nf2-extrapolate`

Run one YAML-configured extrapolation.


| Option | Default | Description |
| --- | --- | --- |
| --config | required | Path to a YAML configuration file. |
| --<placeholder> | optional | Any command-line option matching a <<placeholder>> in the YAML fills that value. |


## `nf2-extrapolate-series`

Run a YAML-configured time series.


| Option | Default | Description |
| --- | --- | --- |
| --config | required | Path to a series YAML configuration file. |
| --<placeholder> | optional | Fills matching <<placeholder>> values, including glob patterns for series files. |


## `nf2-download`

Download supported JSOC/HMI data sources.


| Option | Default | Description |
| --- | --- | --- |
| --source | hmi_sharp | One of hmi_sharp, hmi_synoptic, hmi_full_disk. |
| --download_dir | required | Output directory. |
| --email | required | JSOC export email address. |
| --sharp_num | none | HARP/SHARP number for hmi_sharp. |
| --noaa_num | none | NOAA active-region number, resolved to HARP when needed. |
| --t_start | source dependent | Start time for hmi_sharp and hmi_full_disk. |
| --t_end | none | Optional end time. |
| --cadence | 720s | Cadence for time-range downloads. |
| --series | source default | JSOC series override. |
| --segments | source default | Comma-separated JSOC segments. |
| --carrington_rotation | none | Required for hmi_synoptic. |
| --carrington_rotation_end | same as start | Optional inclusive final Carrington rotation. |
| --include_mr_polfil | false | Also download hmi.synoptic_mr_polfil_720s for synoptic workflows. |
| --no_convert_ptr | false | Disable HmiB2ptr conversion for full-disk vector data. |
| --keep_coordinates | false | Keep generated latitude/longitude files after full-disk conversion. |


## `nf2-export`

Export NF2 results to exchange formats.


| Option | Default | Description |
| --- | --- | --- |
| nf2_path | required | One or more .nf2 paths or glob patterns. |
| --format | vtk | One of vtk, npz, hdf5, h5, fits. |
| --out | none | Output file for a single input. |
| --out-dir | current directory | Output directory for multiple inputs. |
| --Mm_per_pixel | checkpoint default | Cartesian export sampling. |
| --height_range | full height | Cartesian height range in Mm. |
| --x_range | full x | Cartesian x range in Mm. |
| --y_range | full y | Cartesian y range in Mm. |
| --metrics | j | Derived quantities to include. |
| --overwrite | false | Replace existing series export files. |


## `nf2-metrics`

Print standard NLFF quality metrics.


| Option | Default | Description |
| --- | --- | --- |
| nf2_path | required | Path to extrapolation_result.nf2. |
| --device | auto | PyTorch evaluation device. |
| --progress | false | Show evaluation progress. |
| --batch_size | output default | Evaluation batch size. |
| --Mm_per_pixel | checkpoint default | Cartesian sampling. |
| --height_range | full height | Cartesian height range in Mm. |
| --x_range | full x | Cartesian x range in Mm. |
| --y_range | full y | Cartesian y range in Mm. |
| --spherical_sampling | 32 64 128 | Spherical radial, latitude, longitude sample counts. |
| --radius_range | checkpoint range | Spherical radius range in solar radii. |
| --latitude_range | -90 90 | Spherical latitude range in degrees. |
| --longitude_range | 0 360 | Spherical longitude range in degrees. |

