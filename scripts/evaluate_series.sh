module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/pub_NF2
python3 -i -m nf2.evaluation.series /gpfs/gpfs0/robert.jarolim/nf2/377/series --result_path /gpfs/gpfs0/robert.jarolim/nf2/377/evaluation --strides 1 --flare_list /gpfs/gpfs0/robert.jarolim/data/goes_flares_integrated.csv
