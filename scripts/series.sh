#!/bin/bash -l

#PBS -N series
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=4:mem=24gb
#PBS -l walltime=24:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/NF2

#python3 -m nf2.extrapolate --config config/sharp/377.yaml
#python3 -m nf2.extrapolate_series --config config/sharp/377_series.yaml

python3 -m nf2.extrapolate_series --config config/sharp/13664_series_part2.yaml
