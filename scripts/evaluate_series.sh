#!/bin/bash
#SBATCH --job-name=nf2                       # Job name
#SBATCH --partition=gpu_devel                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks-per-node=4                          # Run 12 tasks
#SBATCH --mem=24000                          # Job memory request in Megabytes
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --time=12:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/nf2/logs/nf2_train_%j.log     # Standard output and error log


module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/pub_NF2
python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/401/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/401/evaluation --strides 2 --add_flares
#python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/6975/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/6975/evaluation --strides 4 --add_flares
