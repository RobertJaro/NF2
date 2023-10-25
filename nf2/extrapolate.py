import argparse
import json
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from nf2.train.module import NF2Module, save
from nf2.train.data_loader import NumpyDataModule, SOLISDataModule, FITSDataModule, AnalyticDataModule, SHARPDataModule

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
else:
    raise NotImplementedError(f'Unknown data loader {args.data["type"]}')

validation_settings = {'cube_shape': data_module.cube_dataset.coords_shape,
                       'gauss_per_dB': args.data["b_norm"],
                       'Mm_per_ds': args.data["Mm_per_pixel"] * args.data["spatial_norm"]}

nf2 = NF2Module(validation_settings, **args.model, **args.training)

config = {'data': args.data, 'model': args.model, 'training': args.training}
save_callback = LambdaCallback(
    on_validation_end=lambda *args: save(save_path, nf2.model, data_module, config, nf2.height_mapping_model))
checkpoint_callback = ModelCheckpoint(dirpath=base_path, every_n_train_steps=args.training["validation_interval"],
                                      save_last=True)

torch.set_float32_matmul_precision('medium')  # for A100 GPUs
n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=2,
                  logger=wandb_logger,
                  devices=n_gpus if n_gpus > 0 else None,
                  accelerator='gpu' if n_gpus >= 1 else None,
                  strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                  num_sanity_val_steps=0,
                  val_check_interval=args.training['validation_interval'],
                  gradient_clip_val=0.1,
                  callbacks=[checkpoint_callback, save_callback], )

trainer.fit(nf2, data_module, ckpt_path='last')
save(save_path, nf2.model, data_module, config, height_mapping_model=nf2.height_mapping_model)
# clean up
data_module.clear()


def main(): # workaround for entry_points
    pass