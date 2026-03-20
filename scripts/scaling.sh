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

#python -m nf2.extrapolate --config config/scaling/377.yaml
python -m nf2.extrapolate --config config/scaling/11515.yaml

python3 -m nf2.data.download_range --download_dir /glade/work/rjarolim/data/nf2/11515 --email robert.jarolim@uni-graz.at --noaa_num 11515 --t_start 2012-07-04T14:00:00


#####
# convert
python -m nf2.convert.nf2_to_vtk --nf2_path "/glade/work/rjarolim/nf2/scaling/11515_v01/extrapolation_result.nf2"