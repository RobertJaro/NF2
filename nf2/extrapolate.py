import argparse
import json
import os

import numpy as np
import torch
from astropy.nddata import block_reduce

from nf2.data.loader import load_hmi_data
from nf2.train.trainer import NF2Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--meta_path', type=str, required=False, default=None)
parser.add_argument('--positional_encoding', action='store_true')
parser.add_argument('--use_vector_potential', action='store_true')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

# data parameters
bin = int(args.bin)
spatial_norm = 320 // bin
height = 320 // bin
b_norm = 2500

# model parameters
dim = args.dim

# training parameters
lambda_div = args.lambda_div
lambda_ff = args.lambda_ff
n_gpus = torch.cuda.device_count()
batch_size = int(args.batch_size)
log_interval = args.log_interval
validation_interval = args.validation_interval
potential = args.potential
num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()

base_path = args.base_path
data_path = args.data_path

hmi_cube, error_cube, meta_info = load_hmi_data(data_path)

if 'slice' in args:
    slice = args.slice
    hmi_cube = hmi_cube[slice[0]:slice[1], slice[2]:slice[3]]
    error_cube = error_cube[slice[0]:slice[1], slice[2]:slice[3]]

# bin data
if bin > 1:
    hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
    error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)
# init trainer
trainer = NF2Trainer(base_path, hmi_cube, error_cube, height, spatial_norm, b_norm,
                     meta_info=meta_info, dim=dim,
                     positional_encoding=args.positional_encoding,
                     use_potential_boundary=potential, lambda_div=lambda_div, lambda_ff=lambda_ff,
                     decay_iterations=args.decay_iterations, meta_path=args.meta_path,
                     use_vector_potential=args.use_vector_potential, work_directory=args.work_directory)
trainer.train(args.iterations, batch_size, log_interval, validation_interval, num_workers=num_workers)
