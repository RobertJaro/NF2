import argparse
import json
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from nf2.train.callback import SlicesCallback, BoundaryCallback
from nf2.train.module import NF2Module, save
from nf2.train.data_loader import NumpyDataModule, SOLISDataModule, FITSDataModule, AnalyticDataModule, SHARPDataModule, \
    SphericalDataModule, SynopticDataModule, PotentialTestDataModule, AzimuthDataModule

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

save_path = os.path.join(base_path, 'extrapolation_result.nf2')

# init logging
wandb_id = args.logging['wandb_id'] if 'wandb_id' in args.logging else None
log_model = args.logging['wandb_log_model'] if 'wandb_log_model' in args.logging else False
wandb_logger = WandbLogger(project=args.logging['wandb_project'], name=args.logging['wandb_name'], offline=False,
                           entity=args.logging['wandb_entity'], id=wandb_id, dir=base_path, log_model=log_model)
wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

# restore model checkpoint from wandb
if wandb_id is not None:
    checkpoint_reference = f"{args.logging['wandb_entity']}/{args.logging['wandb_project']}/model-{args.logging['wandb_id']}:latest"
    artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
    artifact.download(root=base_path)
    shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
    args.data['plot_overview'] = False  # skip overview plot for restored model

callbacks = []

if 'work_directory' not in args.data or args.data['work_directory'] is None:
    args.data['work_directory'] = base_path
if args.data["type"] == 'numpy':
    data_module = NumpyDataModule(**args.data)
elif args.data["type"] == 'sharp':
    data_module = SHARPDataModule(**args.data)
elif args.data["type"] == 'fits':
    data_module = FITSDataModule(**args.data)
elif args.data["type"] == 'solis':
    data_module = SOLISDataModule(**args.data)
elif args.data["type"] == 'analytical':
    data_module = AnalyticDataModule(**args.data)
elif args.data["type"] == 'spherical':
    data_module = SphericalDataModule(**args.data, plot_settings=args.plot)
elif args.data["type"] == 'azimuth':
    data_module = AzimuthDataModule(**args.data, plot_settings=args.plot)
    boundary_callback = BoundaryCallback(data_module.cube_shape)
    callbacks += [boundary_callback]
elif args.data["type"] == 'synoptic':
    data_module = SynopticDataModule(**args.data, plot_settings=args.plot)
elif args.data["type"] == 'potential_test':
    data_module = PotentialTestDataModule(**args.data, plot_settings=args.plot)
else:
    raise NotImplementedError(f'Unknown data loader {args.data["type"]}')

plot_slices_callbacks = [SlicesCallback(plot_settings['name'], data_module.slices_datasets[plot_settings['name']].cube_shape)
                         for plot_settings in args.plot if plot_settings['type'] == 'slices']
# boundary_callback = BoundaryCallback(data_module.cube_shape)

validation_settings = {'cube_shape': data_module.cube_dataset.coords_shape,
                       'gauss_per_dB': args.data["b_norm"],
                       'Mm_per_ds': args.data["Mm_per_pixel"] * args.data["spatial_norm"] if "spatial_norm" in args.data else 1,
                       'names': [plot_settings['name'] for plot_settings in args.plot],
                       }

nf2 = NF2Module(validation_settings, **args.model, **args.training)

config = {'data': args.data, 'model': args.model, 'training': args.training}
save_callback = LambdaCallback(
    on_validation_end=lambda *args: save(save_path, nf2.model, data_module, config))
checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                      every_n_train_steps=args.training["validation_interval"] if "validation_interval" in args.training else None,
                                      every_n_epochs=args.training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                                      save_last=True)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()
callbacks += [checkpoint_callback, save_callback, *plot_slices_callbacks]
trainer = Trainer(max_epochs=int(args.training['epochs']) if 'epochs' in args.training else 1,
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  val_check_interval=int(args.training['validation_interval']) if 'validation_interval' in args.training else None,
                  check_val_every_n_epoch=args.training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in args.training else None,
                  gradient_clip_val=0.1,
                  callbacks=callbacks)

trainer.fit(nf2, data_module, ckpt_path='last')
save(save_path, nf2.model, data_module, config)
# clean up
data_module.clear()
