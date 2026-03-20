#!/bin/bash -l

#PBS -N vtk
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=01:00:00
#PBS -l job_priority=economy

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/NF2
# full disk
#python3 -m nf2.evaluation.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2173_revive_v1/extrapolation_result.nf2" --out_path "/glade/work/rjarolim/nf2/spherical/2173_revive_v1/extrapolation_result.vtk"
#python3 -i -m nf2.convert.global.potential_vtk --Br "/glade/work/rjarolim/data/global/synoptic/hmi.synoptic_mr_polfil_720s.2270.Mr_polfil.fits" --out_path "/glade/work/rjarolim/nf2/synoptic/potential" --radius_range 0.999 2.0

# 2106 AR
#python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2106_AR_v02/*.nf2" --overwrite --radius_range 0.999 1.1 --latitude_range 1.7 2.1 --longitude_range 0.3 0.8 --pixels_per_solRad 512

python3 -m nf2.convert.nf2_to_vtk --nf2_path "/glade/work/rjarolim/nf2/sst/13392_1slice_v01/extrapolation_result.nf2" --Mm_per_pixel 0.72 --metrics "j" "b_nabla_bz"

python3 -m nf2.convert.nf2_to_vtk --nf2_path "/glade/work/rjarolim/nf2/sst/13392_v13/extrapolation_result.nf2" --Mm_per_pixel 1.44 --metrics "j" "squashing_factor" "b_nabla_bz"
python3 -m nf2.convert.nf2_to_vtk --nf2_path "/glade/work/rjarolim/nf2/sst/sharp_13392_v01/extrapolation_result.nf2" --Mm_per_pixel 0.72 --metrics "j" "squashing_factor" "b_nabla_bz"



python3 -m nf2.convert.nf2_to_vtk --nf2_path "/glade/work/rjarolim/nf2/sharp/13664/series/20240506_081200_TAI.nf2" --vtk_path "/glade/work/rjarolim/nf2/sharp/13664/evaluation/20240506_081200_TAI.vtk" --Mm_per_pixel 0.72

# full disk 2267
python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/377_v03/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64


python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2173_subframe_v02/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64 --latitude_range 0 1.57 --longitude_range 5.0 8.0 --radians


python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2173_post_eruption_v01/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 128 --latitude_range 0.424 2.938 --longitude_range 1.603 4.116 --radians


python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2154_v01/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64 --latitude_range 0.186 2.699 --longitude_range 2.087 4.915 --radians


# full disk 2283
python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2283_00_v01/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64 --latitude_range 0.398 2.911 --longitude_range 1.673 4.501 --radians

python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/Users/rjarolim/PycharmProjects/NF2/results/2283/extrapolation_2SR.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64 --latitude_range 0.398 2.911 --longitude_range 1.673 4.501 --radians

# full disk 2154
python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2154_v02/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64 --latitude_range 0.186 2.699  --longitude_range 2.087 4.915  --radians

# full disk 2173
python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2173_15_v01/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.5 --pixels_per_solRad 128 --latitude_range 0.424 2.938  --longitude_range 1.446 4.273  --radians


# full 2173
python3 -m nf2.convert.nf2_to_vtk_spherical --nf2_path "/glade/work/rjarolim/nf2/spherical/2173_synoptic_v01/extrapolation_result.nf2" --overwrite --radius_range 0.999 1.3 --pixels_per_solRad 64
