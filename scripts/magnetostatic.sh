#!/bin/bash -l

#PBS -N NF2-magnetostatic
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=2:mem=24gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/NF2

#python -i -m nf2.train.fit_muram_p_profile  --out_path "/glade/work/rjarolim/nf2/magnetostatic/profile" --muram_source_path "/glade/campaign/hao/radmhd/rjarolim/magnetostatic" --iteration 1000000
#python -i -m nf2.train.fit_muram_rho_profile  --out_path "/glade/work/rjarolim/nf2/magnetostatic/profile" --muram_source_path "/glade/campaign/hao/radmhd/rjarolim/magnetostatic" --iteration 1000000

#python -m nf2.train.fit_muram_p_profile  --out_path "/glade/work/rjarolim/nf2/magnetostatic/profile_MFR" --muram_source_path "/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D" --iteration 474000 --base_height 64

#python -m nf2.extrapolate --config config/magnetostatic/force_free.yaml
python -m nf2.extrapolate --config config/magnetostatic/standard.yaml
#python -m nf2.extrapolate --config config/magnetostatic/cutoff.yaml
#python -m nf2.extrapolate --config config/magnetostatic/pressure_boundary.yaml
#python -m nf2.extrapolate --config config/magnetostatic/scaling.yaml
#python -m nf2.extrapolate --config config/magnetostatic/pressure_full_cube.yaml


# VTK
#python -m nf2.convert.nf2_to_vtk --nf2_path "/Users/rjarolim/PycharmProjects/NF2/results/magnetostatic/extrapolation_result.nf2" --out "/Users/rjarolim/PycharmProjects/NF2/results/magnetostatic/muram.vtk" --Mm_per_pixel 1.44 --metrics "j"


python -i -m nf2.evaluation.muram.compare_pressure --nf2_path "/glade/work/rjarolim/nf2/magnetostatic/standard_v02/extrapolation_result.nf2" --out_path "/glade/work/rjarolim/nf2/magnetostatic/standard_v02/evaluation" --muram_source_path "/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D" --muram_iteration 474000

python -i -m nf2.evaluation.muram.compare_magnetic_energy --nf2_path_magnetostatic "/glade/campaign/hao/radmhd/rjarolim/magnetostatic/transfer_boundary_cutoff.nf2" --nf2_path_force_free "/glade/campaign/hao/radmhd/rjarolim/magnetostatic/force_free.nf2" --out_path "/glade/work/rjarolim/nf2/magnetostatic/evaluation" --muram_source_path "/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D" --muram_iteration 474000

