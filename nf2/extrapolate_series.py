import argparse
import glob
import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from nf2.module import NF2Module, save
from nf2.train.data_loader import SHARPSeriesDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

base_path = args.base_path
os.makedirs(base_path, exist_ok=True)

# init logging
wandb_id = args.logging['wandb_id'] if 'wandb_id' in args.logging else None
wandb_logger = WandbLogger(project=args.logging['wandb_project'], name=args.logging['wandb_name'], offline=False,
                           entity=args.logging['wandb_entity'], id=wandb_id, dir=base_path, log_model='all')
wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

# general training parameters
torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()

data_path = args.data['paths']
hmi_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

data_paths = list(zip(hmi_p_files, err_p_files, hmi_t_files, err_t_files, hmi_r_files, err_r_files))

# reload model training
ckpts = sorted(glob.glob(os.path.join(base_path, '*.nf2')))
meta_path = ckpts[-1] if len(ckpts) > 0 else args.meta_path  # reload last savepoint
data_paths = data_paths[len(ckpts):]  # select remaining extrapolations

# initial training
data_module = SHARPSeriesDataModule(data_paths, **args.data)
validation_settings = {'cube_shape': data_module.cube_dataset.coords_shape,
                       'gauss_per_dB': args.data["b_norm"],
                       'Mm_per_ds': args.data["Mm_per_pixel"] * args.data["spatial_norm"]}
nf2 = NF2Module(validation_settings, meta_path=meta_path, **args.model, **args.training)

# callback
save_callback = LambdaCallback(on_validation_epoch_end=lambda *args: save(
    os.path.join(base_path, os.path.basename(data_paths[nf2.current_epoch][0]).split('.')[-3] + '.nf2'),
    nf2.model, data_module, nf2.height_mapping_model))

trainer = Trainer(max_epochs=len(data_paths),
                  logger=wandb_logger,
                  devices=n_gpus,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0, callbacks=[save_callback],
                  gradient_clip_val=0.1, reload_dataloaders_every_n_epochs=1)
trainer.fit(nf2, data_module)
