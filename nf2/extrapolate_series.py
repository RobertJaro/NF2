import argparse
import glob
import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nf2.train.callback import SlicesCallback
from nf2.train.data_loader import SHARPSeriesDataModule, SphericalSeriesDataModule, SynopticSeriesDataModule
from nf2.train.module import NF2Module, save

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

data_path = args.data['paths']
if isinstance(data_path, str):
    p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
    t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
    r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
    err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
    err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
    err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

    data_paths = list(zip(p_files, err_p_files, t_files, err_t_files, r_files, err_r_files))
    data_paths = [{'Bp': d[0], 'Bp_err': d[1], 'Bt': d[2], 'Bt_err': d[3], 'Br': d[4], 'Br_err': d[5]} for d in
                  data_paths]
elif isinstance(data_path, dict):
    p_files = sorted(glob.glob(data_path['Bp']))  # x
    t_files = sorted(glob.glob(data_path['Bt']))  # y
    r_files = sorted(glob.glob(data_path['Br']))  # z
    err_p_files = sorted(glob.glob(data_path['Bp_err'])) if 'Bp_err' in data_path else None  # x
    err_t_files = sorted(glob.glob(data_path['Bt_err'])) if 'Bt_err' in data_path else None  # y
    err_r_files = sorted(glob.glob(data_path['Br_err'])) if 'Br_err' in data_path else None  # z

    if err_p_files is not None and err_t_files is not None and err_r_files is not None:
        assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(err_r_files), \
            f'Number of files in data path {data_path} does not match'
        data_paths = list(zip(p_files, err_p_files, t_files, err_t_files, r_files, err_r_files))
        data_paths = [{'Bp': d[0], 'Bp_err': d[1],
                       'Bt': d[2], 'Bt_err': d[3],
                       'Br': d[4], 'Br_err': d[5]} for d in data_paths]
    else:
        assert len(p_files) == len(t_files) == len(r_files), \
            f'Number of files in data path {data_path} does not match'
        data_paths = list(zip(p_files, t_files, r_files))
        data_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in data_paths]
else:
    raise NotImplementedError(f'Unknown data path type {type(data_path)}')

# reload model training
ckpts = sorted(glob.glob(os.path.join(base_path, '*.nf2')))
ckpt_path = 'last' if len(ckpts) > 0 else args.meta_path  # reload last savepoint
data_paths = data_paths[len(ckpts):]  # select remaining extrapolations

if 'work_directory' not in args.data or args.data['work_directory'] is None:
    args.data['work_directory'] = base_path
if args.data["type"] == 'sharp':
    data_module = SHARPSeriesDataModule(data_paths, **args.data)
elif args.data["type"] == 'spherical':
    data_module = SphericalSeriesDataModule(data_paths, **args.data, plot_settings=args.plot)
elif args.data["type"] == 'synoptic':
    data_module = SynopticSeriesDataModule(data_paths, **args.data, plot_settings=args.plot)
else:
    raise NotImplementedError(f'Unknown data loader {args.data["type"]}')

plot_slices_callbacks = [
    SlicesCallback(plot_settings['name'], data_module.slices_datasets[plot_settings['name']].cube_shape)
    for plot_settings in args.plot if plot_settings['type'] == 'slices']

validation_settings = {'cube_shape': data_module.cube_dataset.coords_shape,
                       'gauss_per_dB': args.data["b_norm"],
                       'Mm_per_ds': args.data["Mm_per_pixel"] * args.data[
                           "spatial_norm"] if "spatial_norm" in args.data else 1,
                       'names': [plot_settings['name'] for plot_settings in args.plot],
                       }

nf2 = NF2Module(validation_settings, **args.model, **args.training)

reload_dataloaders_every_n_epochs = args.training[
    'reload_dataloaders_every_n_epochs'] if 'reload_dataloaders_every_n_epochs' in args.training else 1

# callback
config = {'data': args.data, 'model': args.model, 'training': args.training}
save_callback = LambdaCallback(on_train_epoch_end=lambda *args: save(
    os.path.join(base_path, os.path.basename(data_module.current_files['Br']).split('.')[-3] + '.nf2'),
    nf2.model, data_module, config))

checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_epochs=args.training[
                                          'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else 1,
                                      save_last=True)

# general training parameters
torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()

trainer = Trainer(max_epochs=-1,
                  logger=wandb_logger,
                  devices=n_gpus,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0, callbacks=[save_callback, checkpoint_callback, *plot_slices_callbacks],
                  gradient_clip_val=0.1, reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
                  check_val_every_n_epoch=args.training[
                      'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else 1, )
trainer.fit(nf2, data_module, ckpt_path=ckpt_path)
