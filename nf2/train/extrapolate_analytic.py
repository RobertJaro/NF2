import argparse
import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nf2.module import NF2Module, save
from nf2.loader.analytical import AnalyticDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
parser.add_argument('--num_workers', type=int, required=False, default=4)
parser.add_argument('--meta_path', type=str, required=False, default=None)
parser.add_argument('--positional_encoding', action='store_true')
parser.add_argument('--use_vector_potential', action='store_true')
parser.add_argument('--use_height_mapping', action='store_true')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

# data parameters
spatial_norm = 64.
height = 64 # 72
b_norm = 300
# model parameters
dim = args.dim
# training parameters
lambda_b = args.lambda_b
lambda_div = args.lambda_div
lambda_ff = args.lambda_ff
n_gpus = torch.cuda.device_count()
batch_size = int(args.batch_size)
validation_interval = args.validation_interval
num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
use_vector_potential = args.use_vector_potential
positional_encoding = args.positional_encoding
use_height_mapping = args.use_height_mapping
iterations = args.iterations

base_path = args.base_path
work_directory = args.work_directory
work_directory = base_path if work_directory is None else work_directory

os.makedirs(base_path, exist_ok=True)
os.makedirs(work_directory, exist_ok=True)

save_path = os.path.join(base_path, 'extrapolation_result.nf2')

slice = args.slice if 'slice' in args else None

# INIT TRAINING
logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, offline=False, entity="robert_jarolim")
logger.experiment.config.update(vars(args))

data_module = AnalyticDataModule(args.case, args.height_slices, height, spatial_norm, b_norm,
                                 work_directory, batch_size, batch_size * 2, iterations, num_workers,
                                 boundary=args.boundary, potential_strides=1, tau_surfaces=args.tau_surfaces)

validation_settings = {'cube_shape': data_module.cube_shape,
                       'gauss_per_dB': b_norm,
                       'Mm_per_ds': 320 * 360e-3}
nf2 = NF2Module(validation_settings, dim, lambda_b, lambda_div, lambda_ff,
                meta_path=args.meta_path, positional_encoding=positional_encoding,
                use_vector_potential=use_vector_potential, use_height_mapping=use_height_mapping, )

save_callback = LambdaCallback(
    on_validation_end=lambda *args: save(save_path, nf2.model, data_module, nf2.height_mapping_model))
checkpoint_callback = ModelCheckpoint(dirpath=base_path, monitor='train/loss',
                                      every_n_train_steps=validation_interval, save_last=True)

trainer = Trainer(max_epochs=1,
                  logger=logger,
                  devices=n_gpus,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  val_check_interval=validation_interval,
                  gradient_clip_val=0.1,
                  callbacks=[checkpoint_callback, save_callback])

trainer.fit(nf2, data_module, ckpt_path='last')
save(save_path, nf2.model, data_module, height_mapping_model=nf2.height_mapping_model)
# clean up
data_module.clear()
