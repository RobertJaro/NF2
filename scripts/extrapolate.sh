#!/bin/bash -l

#PBS -N extrapolate
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/NF2

#python -m nf2.extrapolate --config config/sharp/iswat_AR12760_scaling.yaml

python -m nf2.extrapolate --config config/sharp/377.yaml

#python -m nf2.extrapolate --config config/multi_height/muram_disambiguation.yaml
#python -m nf2.extrapolate --config config/sharp/377.yaml
#python -m nf2.extrapolate --config config/multi_height/muram_xflare.yaml
#python -m nf2.extrapolate --config config/sharp/13664_ensemble/run04.yaml
#python -m nf2.extrapolate --config config/sharp/13229.yaml

#python -m nf2.extrapolate --config config/spherical/2173_post_eruptoin.yaml
#python -m nf2.extrapolate --config config/spherical/377_spherical.yaml
#python -m nf2.extrapolate --config config/spherical/377_subframe.yaml
#python -m nf2.extrapolate --config config/spherical/377_embedded.yaml
#python -m nf2.extrapolate --config config/spherical/377_potential.yaml
#python -m nf2.extrapolate --config config/spherical/2267_subframe.yaml
#python -m nf2.extrapolate --config config/sst/7310.yaml
#python -m nf2.extrapolate --config config/spherical/2283_04_v02.yaml
#python -m nf2.extrapolate --config config/spherical/2154.yaml
#python -m nf2.extrapolate --config config/spherical/2173_full.yaml
#python -m nf2.extrapolate --config config/spherical/2173.yaml


#python -m nf2.extrapolate --config config/disambiguation/muram_mfr_ambiguous.yaml
#python -m nf2.extrapolate --config config/disambiguation/muram_mfr_multi.yaml
#python -m nf2.extrapolate --config config/disambiguation/muram_mfr.yaml

#python -m nf2.extrapolate --config config/multi_height/muram_mfr.yaml
#python -m nf2.extrapolate --config config/multi_height/muram_mfr_single_height.yaml
# python -m nf2.extrapolate --config config/multi_height/muram_sma.yaml
#python -m nf2.extrapolate --config config/multi_height/muram_sma_single_height.yaml

#python -m nf2.extrapolate --config config/sharp/377_magnetostatic.yaml
#python -m nf2.extrapolate --config config/sharp/377_magnetostatic_v2.yaml