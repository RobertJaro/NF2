import argparse
import os
import shutil

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from nf2.loader.analytical import AnalyticDataModule
from nf2.loader.fits import FITSDataModule
from nf2.loader.general import NumpyDataModule
from nf2.loader.muram import MURaMDataModule
from nf2.loader.spherical import SphericalDataModule
from nf2.loader.vsm import VSMDataModule
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

save_path = os.path.join(base_path, 'extrapolation_result.nf2')

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

if args.data["type"] == 'numpy':
    data_module = NumpyDataModule(**args.data)
elif args.data["type"] == 'fits':
    data_module = FITSDataModule(**args.data)
elif args.data["type"] == 'solis':
    data_module = VSMDataModule(**args.data)
elif args.data["type"] == 'analytical':
    data_module = AnalyticDataModule(**args.data)
elif args.data["type"] == 'spherical':
    data_module = SphericalDataModule(**args.data)
elif args.data["type"] == 'muram':
    data_module = MURaMDataModule(**args.data)
else:
    raise NotImplementedError(f'Unknown data loader {args.data["type"]}')

# initialize callbacks
callbacks = load_callbacks(data_module)

nf2 = NF2Module(data_module.validation_dataset_mapping, data_module.config,
                model_kwargs=args.model, **args.training)

config = {'data': args.data, 'model': args.model, 'training': args.training}
save_callback = LambdaCallback(
    on_validation_end=lambda *args: save(save_path, nf2, data_module, config))
checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_train_steps=args.training[
                                          "validation_interval"] if "validation_interval" in args.training else None,
                                      every_n_epochs=args.training[
                                          'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                                      save_last=True)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()
callbacks += [checkpoint_callback, save_callback]
trainer = Trainer(max_epochs=int(args.training['epochs']) if 'epochs' in args.training else 1,
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  val_check_interval=int(
                      args.training['validation_interval']) if 'validation_interval' in args.training else None,
                  check_val_every_n_epoch=args.training[
                      'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                  gradient_clip_val=0.1,
                  callbacks=callbacks)

trainer.fit(nf2, data_module, ckpt_path='last')
save(save_path, nf2, data_module, config)
# clean up
data_module.clear()
