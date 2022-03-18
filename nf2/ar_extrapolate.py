import os

import argparse
import glob
import json

import numpy as np
from astropy.nddata import block_reduce
from sunpy.map import Map

from nf2.train.trainer import NF2Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--meta_path', type=str, required=False, default=None)
parser.add_argument('--positional_encoding', type=bool, required=False, default=False)
parser.add_argument('--n_samples_epoch', type=int, required=False, default=None)
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

# data parameters
bin = args.bin
spatial_norm = 320 // bin
height = 320 // bin
b_norm = 2500
# model parameters
dim = args.dim
# training parameters
lambda_div = args.lambda_div
lambda_ff = args.lambda_ff
epochs = args.epochs
decay_epochs = args.decay_epochs
batch_size = int(args.batch_size)
n_samples_epoch = int(batch_size * 100) if args.n_samples_epoch is None else args.n_samples_epoch
log_interval = args.log_interval
validation_interval = args.validation_interval
potential = args.potential

base_path = args.base_path
data_path = args.data_path
if isinstance(data_path, str):
    hmi_p = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))[0]  # x
    hmi_t = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))[0]  # y
    hmi_r = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))[0]  # z
    err_p = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))[0]  # x
    err_t = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))[0]  # y
    err_r = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))[0]  # z
else:
    hmi_p, err_p, hmi_r, err_r, hmi_t, err_t = data_path
# laod maps
hmi_cube = np.stack([Map(hmi_p).data, -Map(hmi_t).data, Map(hmi_r).data]).transpose()
error_cube = np.stack([Map(err_p).data, Map(err_t).data, Map(err_r).data]).transpose()
if 'slice' in args:
    slice = args.slice
    hmi_cube = hmi_cube[slice[0]:slice[1], slice[2]:slice[3]]
    error_cube = error_cube[slice[0]:slice[1], slice[2]:slice[3]]

# bin data
if bin > 1:
    hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
    error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)
# init trainer
trainer = NF2Trainer(base_path, hmi_cube, error_cube, height, spatial_norm, b_norm, dim, positional_encoding=args.positional_encoding,
                     potential_boundary=potential, lambda_div=lambda_div, lambda_ff=lambda_ff,
                     decay_epochs=decay_epochs, num_workers=args.num_workers, meta_path=args.meta_path)
trainer.train(epochs, batch_size, n_samples_epoch, log_interval, validation_interval)
