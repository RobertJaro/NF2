# Neural Network Force-Free magnetic field extrapolation - NF2
<img src="https://github.com/RobertJaro/NF2/blob/main/images/logo.jpg" width="150" height="150">

# [Usage](#usage) --- [Paper](#paper) --- [Data](#data)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/example/extrapolation.ipynb)


## Abstract

While the photospheric magnetic field of our Sun is routinely measured, its extent into the upper atmosphere remains elusive.
We present a novel approach for coronal magnetic field extrapolation, using a neural network that integrates observational data and the physical force-free magnetic field model. 
Our method flexibly finds a trade-off between the observation and force-free magnetic field assumption, improving the understanding of the connection between the observation and the underlying physics.
We utilize meta-learning concepts to simulate the evolution of active region NOAA 11158. Our simulation of 5 days of observations at full cadence, requires less than 13 hours of total computation time, enabling real-time force-free magnetic field extrapolations. 
A systematic comparison of the time evolution of free magnetic energy and magnetic helicity in the coronal volume, as well as comparison to EUV observations demonstrates the validity of our approach. The obtained temporal and spatial depletion of free magnetic energy unambiguously relates to the observed flare activity.

## Usage

NF2 can be used as framework to download SHARP data, perform extrapolations and to verify the results. 
The colab notebook provides a basic example that can be adjusted for arbitrary active regions.

### Scripts (command line)

Before you start download and install the requirements.txt in your Python3 environment.
Make sure that PyTorch is properly installed if you use a GPU environment.

```
pip install git+https://github.com/RobertJaro/NF2.git
```

In the config file you have to specify the path to your data. You need the three vector components (Bp, Br, Bt) and their corresponding errors.
The base path defines where the results and the trained model are stored. The bin defines spatial reduction of the input data (bin 2 is recommended for SHARP data).
Other parameters are used for model training and can be set to their default value.

Example for SHARP 377 (hmi_7115.json):
```json
{
  "base_path": "/<<your path>>/hmi_377",
  "logging": {
    "wandb_entity": "<<your wandb username>>",
    "wandb_project": "<<your project name>>",
    "wandb_name": "hmi_377",
    "wandb_id": null
  },
  
  "data": {
    "type": "sharp",
    "data_path": [
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Bp.fits",
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Bp_err.fits",
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Bt.fits",
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Bt_err.fits",
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Br.fits",
    "/<<your path>>/hmi.sharp_cea_720s.377.20110212_000000_TAI.Br_err.fits"
    ],
    "bin": 2,
    "height_mapping": {"z":  [0.000]},
    "Mm_per_pixel": 0.72,
    "boundary": {"type":  "potential", "strides":  4},
    "height": 160,
    "b_norm": 2500,
    "spatial_norm": 160,
    "batch_size": {"boundary":  1e4, "random":  2e4},
    "iterations": 1e5,
    "work_directory": null,
    "num_workers": 8
  },

  "model": {
    "dim": 256
  },

  "training": {
    "lambda_b": {"start": 1e3, "end": 1, "iterations" : 5e4},
    "lambda_div": 1e-1,
    "lambda_ff": 1e-1,
    "lambda_height_reg": 1e-3,
    "validation_interval": 1e4,
    "lr_params": {"start": 5e-4, "end": 5e-5, "decay_iterations": 1e5}
  }
}
```

Training requires about 1 hour on a single V100 GPU.

```
python -m nf2.extrapolate --config <<your path>>/hmi_7115.json
```

For the extrapolations of time series we can use the model weights of the previous time step. A single time step can then be performed in a few minutes.
For this we also create a configuration that specifies our data.

Example for the time series of SHARP 377 (hmi_377_series.json):
```json
{
  "base_path": "/<<your path>>/hmi_series_377",
  "meta_path": "/<<your previous training path>>/extrapolation_result.nf2",
  "logging": {
    "wandb_entity": "<<your wandb username>>",
    "wandb_project": "<<your project name>>",
    "wandb_name": "hmi_series_377",
    "wandb_id": null
  },
  
  "data": {
    "type": "sharp",
    "paths": "/<<your path>>/",
    "bin": 2,
    "height_mapping": {"z":  [0.000]},
    "Mm_per_pixel": 0.72,
    "boundary": {"type":  "potential", "strides":  4},
    "height": 160,
    "b_norm": 2500,
    "spatial_norm": 160,
    "batch_size": {"boundary":  1e4, "random":  2e4},
    "iterations": 2e3,
    "work_directory": null,
    "num_workers": 12
  },

  "model": {
    "dim": 256
  },

  "training": {
    "lambda_b": 1,
    "lambda_div": 1e-1,
    "lambda_ff": 1e-1,
    "lr_params": 5e-4,
    "check_val_every_n_epoch": 100
  }
}
```



``` 
python -m nf2.extrapolate_series --config <<your path>>/hmi_377_series.json
```

### Visualization

We recommend Paraview to visualize the results.
https://www.paraview.org/download/ 

![](images/paraview.jpeg)

NF2 models can be converted to VTK files that can be used by Paraview.

``` 
python -m nf2.evaluation.convert_vtk_file <<path to your nf2 file>> <<path to the output vtk file>>
```

## Paper

accepted; Nature Astronomy (Jarolim et al. 2023)

Preprint available: https://doi.org/10.21203/rs.3.rs-1415262/v1

## Data
All our simulation results are publicly available (parameter variation, time series, 66 individual active regions).

http://kanzelhohe.uni-graz.at/nf2/


