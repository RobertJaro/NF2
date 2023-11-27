#!/bin/bash
#SBATCH --job-name=nf2                       # Job name
#SBATCH --partition=gpu                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks-per-node=8                          # Run 12 tasks
#SBATCH --mem=64000                          # Job memory request in Megabytes
#SBATCH --gpus=2                             # Number of GPUs
#SBATCH --time=24:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/nf2/logs/nf2_train_%j.log     # Standard output and error log


module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/pub_NF2
#python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/401/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/401/evaluation --strides 2 --add_flares
#python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/6975/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/6975/evaluation --strides 4 --add_flares
python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/8088/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/8088/evaluation2 --strides 1 --add_flares
