import argparse
import glob
import os
import shutil

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nf2.loader.fits import SHARPSeriesDataModule
from nf2.loader.spherical import SphericalSeriesDataModule
from nf2.loader.synoptic import SynopticSeriesDataModule
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = yaml.safe_load(config)
    for key, value in info.items():
        args.__dict__[key] = value

base_path = args.base_path
os.makedirs(base_path, exist_ok=True)

if not hasattr(args, 'work_directory') or args.work_directory is None:
    setattr(args, 'work_directory', os.path.join(base_path, 'work'))
args.data['work_directory'] = args.work_directory
os.makedirs(args.data['work_directory'], exist_ok=True)

# init logging
wandb_logger = WandbLogger(**args.logging, save_dir=args.data['work_directory'])
wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

# restore model checkpoint from wandb
if 'id' in args.logging:
    checkpoint_reference = f"{args.logging['entity']}/{args.logging['project']}/model-{args.logging['id']}:latest"
    artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
    artifact.download(root=base_path)
    shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
    args.data['plot_overview'] = False  # skip overview plot for restored model

data_path = args.data['data_path']
if isinstance(data_path, str):
    p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
    t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
    r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
    err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
    err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
    err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

    assert len(p_files) == len(t_files) == len(r_files),  f'Number of files in data path {data_path} does not match'
    fits_path = list(zip(p_files, t_files, r_files))
    fits_path = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_path]

    if len(err_p_files) > 0 or len(err_t_files) > 0 or len(err_r_files) > 0:
        assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(err_r_files), \
            f'Number of files in data path {data_path} does not match'
        error_paths = list(zip(err_p_files, err_t_files, err_r_files))
        error_paths = [{'Bp_err': d[0], 'Bt_err': d[1], 'Br_err': d[2]} for d in error_paths]
    else:
        error_paths = None
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
        fits_path = list(zip(p_files, t_files, r_files))
        fits_path = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2], } for d in fits_path]
        error_paths = list(zip(err_p_files, err_t_files, err_r_files))
        error_paths = [{'Bp_err': d[0], 'Bt_err': d[1], 'Br_err': d[2], } for d in error_paths]
    else:
        assert len(p_files) == len(t_files) == len(r_files), \
            f'Number of files in data path {data_path} does not match'
        fits_path = list(zip(p_files, t_files, r_files))
        fits_path = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_path]
        error_paths = None
else:
    raise NotImplementedError(f'Unknown data path type {type(data_path)}')

# reload model training
ckpts = sorted(glob.glob(os.path.join(base_path, '*.nf2')))
ckpt_path = 'last' if len(ckpts) > 0 else args.meta_path  # reload last savepoint
fits_path = fits_path[len(ckpts):]  # select remaining extrapolations
error_paths = error_paths[len(ckpts):] if error_paths is not None else None

if args.data["type"] == 'sharp':
    data_module = SHARPSeriesDataModule(fits_path, error_paths=error_paths, **args.data)
elif args.data["type"] == 'spherical':
    data_module = SphericalSeriesDataModule(fits_path, **args.data, plot_settings=args.plot)
elif args.data["type"] == 'synoptic':
    data_module = SynopticSeriesDataModule(fits_path, **args.data, plot_settings=args.plot)
else:
    raise NotImplementedError(f'Unknown data loader {args.data["type"]}')

callbacks = load_callbacks(data_module)

nf2 = NF2Module(data_module.validation_dataset_mapping, model_kwargs=args.model, **args.training)

reload_dataloaders_every_n_epochs = args.training[
    'reload_dataloaders_every_n_epochs'] if 'reload_dataloaders_every_n_epochs' in args.training else 1

# callback
config = {'data': args.data, 'model': args.model, 'training': args.training}
save_callback = LambdaCallback(
    on_train_epoch_end=lambda *args:
    save(os.path.join(base_path, data_module.current_id + '.nf2'),
         nf2.model, data_module, config))

checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_epochs=args.training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else 1,
                                      save_last=True)

# general training parameters
torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()

trainer = Trainer(max_epochs=-1,
                  logger=wandb_logger,
                  devices=n_gpus,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0, callbacks=[save_callback, checkpoint_callback, *callbacks],
                  gradient_clip_val=0.1, reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
                  check_val_every_n_epoch=args.training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else 1, )
trainer.fit(nf2, data_module, ckpt_path=ckpt_path)


