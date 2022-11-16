import argparse
import glob
import json
import os

import numpy as np
import torch
from astropy.nddata import block_reduce
from tqdm import tqdm

from nf2.data.loader import load_hmi_data
from nf2.train.trainer import NF2Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
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
iterations = args.iterations
decay_iterations = None
n_gpus = torch.cuda.device_count()
batch_size = int(args.batch_size)
log_interval = args.log_interval
validation_interval = args.validation_interval
potential = args.potential
num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()

series_base_path = args.base_path
data_path = args.data_path
meta_path = args.meta_path

# find files
hmi_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

# create series
for hmi_p, hmi_t, hmi_r, err_p, err_t, err_r in tqdm(zip(hmi_p_files, hmi_t_files, hmi_r_files,
                                                    err_p_files, err_t_files, err_r_files), desc='Series', total=len(hmi_p_files)):
    file_id = os.path.basename(hmi_p).split('.')[3]
    base_path = os.path.join(series_base_path, '%s_dim%d_bin%d_pf%s_ld%s_lf%s' % (
        file_id, dim, bin, str(potential), lambda_div, lambda_ff))

    # check if finished
    final_model_path = os.path.join(base_path, 'final.pt')
    if os.path.exists(final_model_path):
        meta_path = final_model_path
        continue

    # load data cube
    hmi_cube, error_cube, meta_info = load_hmi_data([hmi_p, err_p, hmi_r, err_r, hmi_t, err_t])

    if 'slice' in args:
        slice = args.slice
        hmi_cube = hmi_cube[slice[0]:slice[1], slice[2]:slice[3]]
        error_cube = error_cube[slice[0]:slice[1], slice[2]:slice[3]]

    # bin data
    if bin > 1:
        hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
        error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)
    # init trainer
    trainer = NF2Trainer(base_path, hmi_cube, error_cube, height, spatial_norm, b_norm, dim=dim,
                         lambda_div=lambda_div, lambda_ff=lambda_ff,
                         meta_path=meta_path, use_potential_boundary=potential, decay_iterations=decay_iterations,
                         use_vector_potential=args.use_vector_potential, positional_encoding=args.positional_encoding,
                         work_directory=args.work_directory, meta_info=meta_info)

    trainer.train(iterations, batch_size, log_interval, validation_interval)
    meta_path = final_model_path
