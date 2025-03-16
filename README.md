# Neural Network Force-Free magnetic field extrapolation - NF2
<img src="https://github.com/RobertJaro/NF2/blob/main/images/logo.jpg" width="150" height="150">

# [Usage](#usage) --- [Paper](#publications) --- [Data](#data)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertJaro/NF2/blob/main/notebooks/NF2_SHARP_extrapolation.ipynb)


## Abstract

While the photospheric magnetic field of our Sun is routinely measured, its extent into the upper atmosphere remains elusive.
We present a novel approach for coronal magnetic field extrapolation, using a neural network that integrates observational data and the physical force-free magnetic field model. 
Our method flexibly finds a trade-off between the observation and force-free magnetic field assumption, improving the understanding of the connection between the observation and the underlying physics.
We utilize meta-learning concepts to simulate the evolution of active region NOAA 11158. Our simulation of 5 days of observations at full cadence, requires less than 13 hours of total computation time, enabling real-time force-free magnetic field extrapolations. 
A systematic comparison of the time evolution of free magnetic energy and magnetic helicity in the coronal volume, as well as comparison to EUV observations demonstrates the validity of our approach. The obtained temporal and spatial depletion of free magnetic energy unambiguously relates to the observed flare activity.

## Usage

NF2 can be used as framework to download SHARP data, perform extrapolations and to verify the results. 
The colab notebook provides a basic example that can be adjusted for arbitrary active regions.
For local usage configuration files can be used to perform extrapolations of single active regions or series.

## Installation

Use pip to install the NF2 package.
```
pip install nf2
```

For the latest version use the following command.
```
pip install git+https://github.com/RobertJaro/NF2@v0.3.0
```

Make sure that PyTorch is installed if you use a GPU environment. Run in python console:
```python
import torch
    
print('PyTorch version:', torch.__version__)
print('GPU available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
```

## Scripts (command line)

NF2 runs can be configured through yaml files.
The method can be used for single active regions or time series.

### Data download

NF2 provides direct data download of SHARP data from JSOC.

Download of single SHARP region:
```
nf2-download --download_dir "<<PATH TO SAVE>>" --email <<YOUR JSOC REGISTERED EMAIL>> --noaa_num 11158 --t_start 2011-02-15T00:00:00
```

Download of SHARP series:
```
nf2-download --download_dir "<<PATH TO SAVE>>" --email <<YOUR JSOC REGISTERED EMAIL>> --noaa_num 11158 --t_start 2011-02-15T00:00:00 --t_end 2011-02-16T00:00:00
```

To use near real-time data add `--series sharp_cea_720s_nrt`

### AR - Extrapolation
The code provides NLFF extrapolations where a trade-off between the observation and the force-free magnetic field assumption is found. The weighting is controlled through the lambda parameters in the configuration file. Increase the force-free lambda to enforce the force-free magnetic field assumption. If solutions closer to the observation are desired, decrease the force-free lambda.

Basic example for NOAA AR 11158 (377.yaml):
```yaml
---
base_path: "<<PATH TO SAVE THE RESULTS>>"
work_directory: "<<OPTIONAL - SCRATCH DIRECTORY>>"
logging:
  project: "<<YOUR WANDB PROJECT>>"
  name: "SHARP 377"
data:
  type: fits
  slices:
    - fits_path:
        Br: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits"
        Bt: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits"
        Bp: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits"
      error_path: # optional
        Br_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits"
        Bt_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits"
        Bp_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
  num_workers: 8
  iterations: 10000
model:
  type: b
  dim: 256
training:
  epochs: 15
  loss_config:
    - type: boundary
      name: boundary
      lambda: {start: 1.0e+3, end: 1.0, iterations: 5.0e+4}
      ds_id: [boundary_01, potential]
    - type: force_free
      lambda: 1.0e-1
    - type: divergence
      lambda: 1.0e-1
```
The training can be started with the following command and requires about 1 hour on a single V100 GPU.
```
nf2-extrapolate --config <<PATH TO CONFIG FILE>>/377.yaml
```

### AR - Divergence free

To enforce a divergence free magnetic field, the magnetic field can be computed through a vector potential.
For this we specify the model type as vector_potential and remove the divergence loss term. The divergence free condition can be kept for monitoring but should be close to zero. 
Extrapolations that use the vector potential will require more computational resources and training time since the optimization is performed through second order derivatives.

Example for a divergence-free extrapolation of NOAA AR 11158 (377_vp.yaml):
```yaml
---
base_path: "<<PATH TO SAVE THE RESULTS>>"
work_directory: "<<OPTIONAL - SCRATCH DIRECTORY>>"
logging:
  project: "<<YOUR WANDB PROJECT>>"
  name: "SHARP 377 - VP"
data:
  type: fits
  slices:
    - fits_path:
        Br: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits"
        Bt: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits"
        Bp: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits"
      error_path: # optional
        Br_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits"
        Bt_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits"
        Bp_err: "<<ABSOLUTE PATH TO DATA>>/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits"
  num_workers: 8
  iterations: 10000
model:
  type: vector_potential
  dim: 256
training:
  epochs: 15
  loss_config:
    - type: boundary
      name: boundary
      lambda: {start: 1.0e+3, end: 1.0, iterations: 5.0e+4}
      ds_id: [boundary_01, potential]
    - type: force_free
      lambda: 1.0e-1
```

```
nf2-extrapolate --config <<PATH TO CONFIG FILE>>/377_vp.yaml
```

### Time series

For the extrapolations of time series we can use the model weights of the previous time step. A single time step can then be performed in a few minutes.
For this we also create a configuration that specifies our data set.

Example for the time series of NOAA 13664 (related to the May 2024 geomagnetic storm; SHARP 11149).

(initial frame - 13664.yaml)
```yaml
---
base_path: "<<PATH TO SAVE THE RESULTS>>/init"
work_directory: "<<OPTIONAL - SCRATCH DIRECTORY>>"
logging:
  project: "<<YOUR WANDB PROJECT>>"
  name: "NOAA 13664 - init"
data:
  type: fits
  slices:
    - fits_path:
        Br: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Br.fits"
        Bt: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Bt.fits"
        Bp: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Bp.fits"
      error_path:
        Br_err: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Br_err.fits"
        Bt_err: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Bt_err.fits"
        Bp_err: "<<ABSOLUTE PATH TO DATA>>//hmi.sharp_cea_720s.11149.20240505_000000_TAI.Bp_err.fits"
  num_workers: 8
  iterations: 10000
model:
  type: vector_potential
  dim: 256
training:
  epochs: 15
  loss_config:
    - type: boundary
      name: boundary
      lambda: {start: 1.0e+3, end: 1.0, iterations: 5.0e+4}
      ds_id: [boundary_01, potential]
    - type: force_free
      lambda: 1.0e-1
```
(series - 13664_series.yaml)

```yaml
---
base_path: "<<PATH TO SAVE THE RESULTS>>/series"
work_directory: "<<OPTIONAL - SCRATCH DIRECTORY>>"
meta_path: "<<PATH TO SAVE THE RESULTS>>/init/last.ckpt"
logging:
  project: "<<YOUR WANDB PROJECT>>"
  name: "NOAA 13664 - series"
data:
  type: sharp
  data_path: "<<ABSOLUTE PATH TO DATA>>"
  num_workers: 8
  iterations: 2.0e+3
model:
  type: vector_potential
  dim: 256
training:
  check_val_every_n_epoch: 5 # validation plots in 1h steps
  loss_config:
    - type: boundary
      name: boundary
      lambda: 1
      ds_id: [boundary_01, potential]
    - type: force_free
      lambda: 1.0e-1
```

Run initial extrapolation and series extrapolation:
``` 
nf2-extrapolate --config <<your path>>/13664.yaml
nf2-extrapolate-series --config <<your path>>/13664_series.yaml
```

### Visualization

We recommend Paraview to visualize the results.
https://www.paraview.org/download/ 

![](images/paraview.jpeg)

NF2 models can be converted to VTK files that can be used by Paraview. 
Full resolution extrapolations typically exceed the memory capacity. 
`--Mm_per_pixel 0.72` can be used to reduce the resolution to bin2 SHARP data (the default SHARP resolution is 0.36 Mm per pixel).

``` 
nf2-to-vtk --nf2_path <<path to your nf2 file>> --out_path <<path to the output vtk file>> --Mm_per_pixel 0.72
```

## Publications

### Original method paper
[Jarolim et al. 2023, Nature Astronomy](https://doi.org/10.1038/s41550-023-02030-9) \
Jarolim, R., Thalmann, J.K., Veronig, A.M. et al. **Probing the solar coronal magnetic field with physics-informed neural networks**. Nat Astron 7, 1171–1179 (2023). https://doi-org.cuucar.idm.oclc.org/10.1038/s41550-023-02030-9

BibTeX:
```
@ARTICLE{2023NatAs...7.1171J,
       author = {{Jarolim}, R. and {Thalmann}, J.~K. and {Veronig}, A.~M. and {Podladchikova}, T.},
        title = "{Probing the solar coronal magnetic field with physics-informed neural networks.}",
      journal = {Nature Astronomy},
         year = 2023,
        month = oct,
       volume = {7},
        pages = {1171-1179},
          doi = {10.1038/s41550-023-02030-9},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023NatAs...7.1171J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

### Multi-height extrapolations
[Jarolim et al. 2024a, ApJL](https://doi.org/10.3847/2041-8213/ad2450) \
Jarolim, R., Tremblay, B., Rempel, M., Molnar, M., Veronig, A. M., Thalmann, J. K., & Podladchikova, T. (2024). **Advancing solar magnetic field extrapolations through multiheight magnetic field measurements**. The Astrophysical Journal Letters, 963(1), L21.

BibTeX:
```
@ARTICLE{2024ApJ...963L..21J,
       author = {{Jarolim}, Robert and {Tremblay}, Benoit and {Rempel}, Matthias and {Molnar}, Momchil and {Veronig}, Astrid M. and {Thalmann}, Julia K. and {Podladchikova}, Tatiana},
        title = "{Advancing Solar Magnetic Field Extrapolations through Multiheight Magnetic Field Measurements}",
      journal = {\apjl},
     keywords = {Solar magnetic fields, Neural networks, Solar corona, 1503, 1933, 1483},
         year = 2024,
        month = mar,
       volume = {963},
       number = {1},
          eid = {L21},
        pages = {L21},
          doi = {10.3847/2041-8213/ad2450},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...963L..21J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

### Divergence-free extrapolations using the vector potential

[Jarolim et al. 2024b, ApJL](https://doi.org/10.3847/2041-8213/ad8914) \
Jarolim, R., Veronig, A. M., Purkhart, S., Zhang, P., & Rempel, M. (2024). **Magnetic Field Evolution of the Solar Active Region 13664**. The Astrophysical Journal Letters, 976(1), L12.

BibTeX:
```
@ARTICLE{2024ApJ...976L..12J,
       author = {{Jarolim}, Robert and {Veronig}, Astrid M. and {Purkhart}, Stefan and {Zhang}, Peijin and {Rempel}, Matthias},
        title = "{Magnetic Field Evolution of the Solar Active Region 13664}",
      journal = {\apjl},
     keywords = {Solar flares, Solar activity, Solar magnetic fields, Solar magnetic reconnection, Magnetohydrodynamical simulations, 1496, 1475, 1503, 1504, 1966, Astrophysics - Solar and Stellar Astrophysics},
         year = 2024,
        month = nov,
       volume = {976},
       number = {1},
          eid = {L12},
        pages = {L12},
          doi = {10.3847/2041-8213/ad8914},
archivePrefix = {arXiv},
       eprint = {2409.08124},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...976L..12J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


### Applications

Purkhart, S., Veronig, A. M., Kliem, B., Jarolim, R., Dissauer, K., Dickson, E. C., ... & Krucker, S. (2024). **Multipoint study of the rapid filament evolution during a confined C2 flare on 28 March 2022, leading to eruption**. Astronomy & Astrophysics, 689, A259.

Purkhart, S., Veronig, A.M., Dickson, E.C., Battaglia, A.F., Krucker, S., Jarolim, R., Kliem, B., Dissauer, K. and Podladchikova, T., 2023. **Multipoint study of the energy release and transport in the 28 March 2022, M4 flare using STIX, EUI, and AIA during the first Solar Orbiter nominal mission perihelion**. Astronomy & Astrophysics, 679, p.A99.

McKevitt, J., Jarolim, R., Matthews, S., Baker, D., Temmer, M., Veronig, A., Reid, H. and Green, L., 2024. **The Link between Nonthermal Velocity and Free Magnetic Energy in Solar Flares**. The Astrophysical Journal Letters, 961(2), p.L29.

Korsós, M.B., Jarolim, R., Erdélyi, R., Veronig, A.M., Morgan, H. and Zuccarello, F., 2024. **First Insights into the Applicability and Importance of Different 3D Magnetic Field Extrapolation Approaches for Studying the Preeruptive Conditions of Solar Active Regions**. The Astrophysical Journal, 962(2), p.171.

## Data
All our simulation results are publicly available (parameter variation, time series, 66 individual active regions).

http://kanzelhohe.uni-graz.at/nf2/


