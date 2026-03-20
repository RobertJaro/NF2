#!/bin/bash
#SBATCH --job-name=nf2                       # Job name
#SBATCH --partition=gpu_devel                      # Queue name
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks-per-node=4                          # Run 12 tasks
#SBATCH --mem=24000                          # Job memory request in Megabytes
#SBATCH --gpus=1                             # Number of GPUs
#SBATCH --time=12:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/gpfs/gpfs0/robert.jarolim/nf2/logs/nf2_download_%j.log     # Standard output and error log


module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/NF2

python3 -i -m nf2.data.download_full_disk --download_dir /glade/work/rjarolim/data/global/fd_2154 --email robert.jarolim@uni-graz.at --carrington_rotation 2154 --t_start 2014-09-02T00:00:00 --t_end 2014-09-03T00:00:00 --convert_ptr --download_synoptic

python3 -m nf2.data.download_full_disk --download_dir /glade/work/rjarolim/data/global/fd_2154 --email robert.jarolim@uni-graz.at --t_start 2014-09-01T00:00:00 --convert_ptr
