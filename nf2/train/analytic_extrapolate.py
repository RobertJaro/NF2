import argparse
import json

import numpy as np

from nf2.evaluation.analytical_field import get_analytic_b_field
from nf2.train.trainer import NF2Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--meta_path', type=str, required=False, default=None)
parser.add_argument('--positional_encoding', action='store_true')
parser.add_argument('--use_vector_potential', action='store_true')
parser.add_argument('--n_samples_epoch', type=int, required=False, default=None)
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

# data parameters
spatial_norm = 64.
height = 72  # 64
b_norm = 300
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

# CASE 1
# hmi_cube = get_analytic_b_field(n = 1, m = 1, l=0.3, psi=np.pi /4)


# CASE 2
hmi_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15, resolution=[80, 80, 72])

error_cube = np.zeros_like(hmi_cube)

# init trainer
trainer = NF2Trainer(base_path, hmi_cube[:, :, 0], error_cube[:, :, 0], height, spatial_norm, b_norm, dim,
                     positional_encoding=args.positional_encoding,
                     use_potential_boundary=potential, lambda_div=lambda_div, lambda_ff=lambda_ff,
                     decay_epochs=decay_epochs, num_workers=args.num_workers, meta_path=args.meta_path,
                     use_vector_potential=args.use_vector_potential)

# coords = [
#     # z
#     np.stack(np.mgrid[:hmi_cube.shape[0], :hmi_cube.shape[1], :1], -1).reshape((-1, 3)),
#     np.stack(np.mgrid[:hmi_cube.shape[0], :hmi_cube.shape[1], hmi_cube.shape[2] - 1:hmi_cube.shape[2]],
#              -1).reshape((-1, 3)),
#     # y
#     np.stack(np.mgrid[:hmi_cube.shape[0], :1, :hmi_cube.shape[2]], -1).reshape((-1, 3)),
#     np.stack(np.mgrid[:hmi_cube.shape[0], hmi_cube.shape[1] - 1:hmi_cube.shape[1], :hmi_cube.shape[2]],
#              -1).reshape((-1, 3)),
#     # x
#     np.stack(np.mgrid[:1, :hmi_cube.shape[1], :hmi_cube.shape[2]], -1).reshape((-1, 3)),
#     np.stack(np.mgrid[hmi_cube.shape[0] - 1:hmi_cube.shape[0], :hmi_cube.shape[1], :hmi_cube.shape[2]],
#              -1).reshape((-1, 3)),
# ]
# values = [hmi_cube[:, :, :1].reshape((-1, 3)), hmi_cube[:, :, -1:].reshape((-1, 3)),
#           hmi_cube[:, :1, :].reshape((-1, 3)), hmi_cube[:, -1:, :].reshape((-1, 3)),
#           hmi_cube[:1, :, :].reshape((-1, 3)), hmi_cube[-1:, :, :].reshape((-1, 3)), ]
#
# coords = np.concatenate(coords).astype(np.float32)
# values = np.concatenate(values).astype(np.float32)
# err = np.zeros_like(values).astype(np.float32)
#
# # normalize B field
# values = Normalize(-b_norm, b_norm, clip=False)(values) * 2 - 1
# values = np.array(values)
#
# boundary_ds = BoundaryDataset(coords, values, err, spatial_norm)
#
# trainer.boundary_ds = boundary_ds

trainer.train(epochs, batch_size, n_samples_epoch, log_interval, validation_interval)
