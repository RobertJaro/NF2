# Neural Network Force-Free magnetic field extrapolation - NF2

NF2 is a framework for neural magnetic-field extrapolation with separate cartesian and spherical geometry implementations sharing one execution, configuration, and export surface.

## Status

The repository is mid-refactor and now uses:

- a shared execution layer in `nf2/core`
- canonical YAML configs in `config/`
- shared geometry adapters in `nf2/geometry`
- shared output/export layers in `nf2/output` and `nf2/export`

## Installation

```bash
pip install nf2
```

For local development:

```bash
pip install -e .
```

PyTorch must be installed in the target runtime environment.

## Main Commands

```bash
nf2-extrapolate --config /path/to/config.yaml
nf2-extrapolate-analytic --config config/analytical/case1.yaml
nf2-extrapolate-series --config /path/to/series.yaml
nf2-export --checkpoint /path/to/result.nf2 --format vtk
```

Analytical Low and Lou examples are available in:

- `config/analytical/case1.yaml`
- `config/analytical/case2.yaml`
- `notebooks/01_Quickstart_Analytical_Extrapolation.ipynb`

User quickstarts are ordered for click-and-go runs:

- `notebooks/01_Quickstart_Analytical_Extrapolation.ipynb`
- `notebooks/02_Quickstart_SHARP_By_Date.ipynb`
- `notebooks/03_Quickstart_Spherical_By_Date.ipynb`

Supported export formats:

- `vtk`
- `hdf5`
- `fits`
- `npz`
- `binary`

## Config Shape

NF2 now uses a canonical config schema:

```yaml
run:
  mode: single
  geometry: cartesian
  output_dir: /path/to/results
  work_dir: /path/to/work

logging:
  project: nf2
  name: demo

data:
  parameters:
    iterations: 10000
    num_workers: 8
  train:
    - type: fits
      fits_path:
        Br: /path/to/Br.fits
        Bt: /path/to/Bt.fits
        Bp: /path/to/Bp.fits
  validation:
    - type: cube
      ds_id: cube

model:
  type: vector_potential
  dim: 256

training:
  epochs: 20

losses:
  - type: force_free
    lambda: 0.1
```

Series configs add:

- `run.mode: series`
- `run.resume_from`
- `data.sequence`

## Documentation

Primary docs are being built in `docs/`.

Start with:

- `docs/installation.md`
- `docs/configuration.md`
- `docs/training.md`
- `docs/export.md`
- `docs/architecture.md`

Local docs build:

```bash
pip install -r docs/requirements.txt
make docs
```

Local preview:

```bash
make docs-serve
```

Read the Docs hosting:

- repository config: [`.readthedocs.yml`](/Users/rjarolim/PycharmProjects/NF2/.readthedocs.yml)
- docs dependencies: [`docs/requirements.txt`](/Users/rjarolim/PycharmProjects/NF2/docs/requirements.txt)
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
[Jarolim et al. 2024, ApJL](https://doi.org/10.3847/2041-8213/ad2450) \
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

### Applications

Purkhart, S., Veronig, A.M., Dickson, E.C., Battaglia, A.F., Krucker, S., Jarolim, R., Kliem, B., Dissauer, K. and Podladchikova, T., 2023. **Multipoint study of the energy release and transport in the 28 March 2022, M4 flare using STIX, EUI, and AIA during the first Solar Orbiter nominal mission perihelion**. Astronomy & Astrophysics, 679, p.A99.

McKevitt, J., Jarolim, R., Matthews, S., Baker, D., Temmer, M., Veronig, A., Reid, H. and Green, L., 2024. **The Link between Nonthermal Velocity and Free Magnetic Energy in Solar Flares**. The Astrophysical Journal Letters, 961(2), p.L29.

Korsós, M.B., Jarolim, R., Erdélyi, R., Veronig, A.M., Morgan, H. and Zuccarello, F., 2024. **First Insights into the Applicability and Importance of Different 3D Magnetic Field Extrapolation Approaches for Studying the Preeruptive Conditions of Solar Active Regions**. The Astrophysical Journal, 962(2), p.171.

## Data
All our simulation results are publicly available (parameter variation, time series, 66 individual active regions).

http://kanzelhohe.uni-graz.at/nf2/
