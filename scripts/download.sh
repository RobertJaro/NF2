module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/pub_NF2
python3 -m nf2.data.download_range --download_dir /gpfs/gpfs0/robert.jarolim/data/nf2/7115 --email robert.jarolim@uni-graz.at --harpnum 7115 --t_start 2017-09-02T00:00:00 --duration 6d
