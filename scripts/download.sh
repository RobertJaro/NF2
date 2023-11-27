module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/pub_NF2
#python3 -m nf2.data.download_sharp --download_dir /gpfs/gpfs0/robert.jarolim/data/nf2/7310 --email robert.jarolim@uni-graz.at --harpnum 7310 --t_start 2018-09-30T09:24:00 --series "sharp_cea_720s"
#python3 -m nf2.data.download_range --download_dir /gpfs/gpfs0/robert.jarolim/data/nf2/7115 --email robert.jarolim@uni-graz.at --harpnum 7115 --t_start 2017-09-02T00:00:00 --duration 6d
python3 -m nf2.data.download_sharp --download_dir /gpfs/gpfs0/robert.jarolim/data/nf2/az_377 --email robert.jarolim@uni-graz.at --harpnum 377 --t_start 2011-02-15T00:00:00 --series "sharp_720s" --segments 'field, inclination, azimuth, disambig'