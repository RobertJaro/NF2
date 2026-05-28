import argparse
import glob
import os
import shutil
from copy import deepcopy

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LambdaCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from nf2.loader.cartesian import CartesianSeriesDataModule
from nf2.loader.spherical import SphericalSeriesDataModule
from nf2.train.callback import AdvanceDatamoduleStep
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save
from nf2.train.util import load_yaml_config, suppress_accumulate_grad_stream_warning


def run(path, data, meta_path, work_path=None, callbacks=None, logging=None, model=None, training=None, losses=None,
        transforms=None, loss_scaling=None, config=None):
    """Run the simulation with the given configuration.

    This function initializes the data loader, the model, the training loop and the logging.
    The simulation is run with the given configuration and the results are stored in path.
    Use the configurations for logging, model and training to overwrite the default settings.

    Args:
        path: Path to the directory where the simulation results are stored.
        data: Dictionary with the data loader configuration.
        meta_path: Path to the initial model checkpoint (run extrapolation first).
        work_path: Path to the directory where the data is stored. If None, the path is used.
        logging: Dictionary with the logging configuration.
        model: Dictionary with the model configuration.
        training: Dictionary with the training configuration.
        config: Dictionary with the configuration for the simulation.
    """
    suppress_accumulate_grad_stream_warning()

    callbacks = [] if callbacks is None else callbacks
    logging = {} if logging is None else logging
    model = {} if model is None else model
    training = {} if training is None else training
    losses = [] if losses is None else losses
    transforms = [] if transforms is None else transforms
    loss_scaling = [] if loss_scaling is None else loss_scaling

    os.makedirs(path, exist_ok=True)

    if work_path is None:
        work_path = os.path.join(path, 'work')
    os.makedirs(work_path, exist_ok=True)
    data_runtime = deepcopy(data)
    data_runtime['work_path'] = work_path  # set work path for data loaders

    # init logging
    wandb_logger = WandbLogger(**logging, save_dir=work_path)
    config_dict = {'path': path, 'work_path': work_path, 'logging': logging,
                   'model': model, 'training': training, 'config': config, 'data': data,
                   'losses': losses, 'transforms': transforms, 'loss_scaling': loss_scaling,
                   'callbacks': callbacks, 'meta_path': meta_path}

    @rank_zero_only
    def _log_hparams(cfg):
        wandb_logger.log_hyperparams(cfg)

    _log_hparams(config_dict)

    # restore model checkpoint from wandb
    if 'id' in logging:
        assert 'entity' in logging and 'project' in logging, '"entity" and "project" must be provided to continue from wandb checkpoint'
        checkpoint_reference = f"{logging['entity']}/{logging['project']}/model-{logging['id']}:latest"
        artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
        artifact.download(root=path)
        shutil.move(os.path.join(path, 'model.ckpt'), os.path.join(path, 'last.ckpt'))
        data_runtime['plot_overview'] = False  # skip overview plot for restored model

    # reload model training
    ckpts = sorted(glob.glob(os.path.join(path, '*.nf2')))
    current_step = len(ckpts)
    last_ckpt_path = os.path.join(path, 'last.ckpt')
    ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    meta_state_path = None if ckpt_path is not None else meta_path

    # initialize data module
    data_module_save_path = os.path.join(work_path, 'data_module.pkl')

    @rank_zero_only
    def _init_data_module():
        data_module_config = deepcopy(data_runtime)
        assert 'type' in data_module_config, 'Data module type must be specified in the configuration'
        data_module_type = data_module_config.pop('type')
        if data_module_type == 'cartesian':
            data_module = CartesianSeriesDataModule(current_step=current_step, **data_module_config)
        elif data_module_type == 'spherical':
            data_module = SphericalSeriesDataModule(current_step=current_step, **data_module_config)
        else:
            raise NotImplementedError(f'Unknown data loader {data_module_type}')
        torch.save(data_module, data_module_save_path)

    _init_data_module()
    # load data module for all ranks
    data_module = torch.load(data_module_save_path, weights_only=False)

    callback_modules = load_callbacks(callbacks, data_module)

    nf2 = NF2Module(data_module.validation_dataset_mapping, data_module.config,
                    model_kwargs=model, loss_config=losses, transforms=transforms, loss_scaling=loss_scaling,
                    meta_path=meta_state_path,
                    lr_params=training.get('optimizer', {"start": 5e-4, "end": 5e-5, "iterations": 1e5}))

    reload_dataloaders_interval = training[
        'reload_dataloaders_every_n_epochs'] if 'reload_dataloaders_every_n_epochs' in training else 1
    max_epochs = data_module.total_steps * reload_dataloaders_interval if ckpt_path is not None else \
        (data_module.total_steps - data_module.step) * reload_dataloaders_interval

    # callback
    config_dict = {'path': path, 'work_path': work_path, 'logging': logging, 'data': data,
                   'model': model, 'training': training, 'config': config, 'losses': losses,
                   'transforms': transforms, 'loss_scaling': loss_scaling, 'callbacks': callbacks,
                   'meta_path': meta_path}
    # Save every training epoch/dataset, independent of the validation cadence.
    save_callback = LambdaCallback(
        on_train_epoch_end=lambda *args:
        save(os.path.join(path, data_module.current_id + '.nf2'),
             nf2, data_module, config_dict))

    checkpoint_callback = ModelCheckpoint(dirpath=path,
                                          every_n_epochs=reload_dataloaders_interval,
                                          save_last=True)

    advance_data_module_callback = AdvanceDatamoduleStep(data_module, reload_dataloaders_interval)

    # general training parameters
    trainer_config = deepcopy(training.get('trainer', {}))
    gradient_clip_val = trainer_config.pop('gradient_clip_val', training.get('gradient_clip_val', 0.1))
    matmul_precision = trainer_config.pop('matmul_precision', training.get('matmul_precision', 'medium'))
    torch.set_float32_matmul_precision(matmul_precision)
    n_gpus = torch.cuda.device_count()
    callback_modules += [checkpoint_callback, save_callback, advance_data_module_callback]
    default_devices = n_gpus if n_gpus > 0 else 1
    default_accelerator = 'gpu' if n_gpus >= 1 else 'cpu'
    default_strategy = DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto'

    val_check_interval = training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training else 1
    trainer = Trainer(max_epochs=max_epochs,
                      logger=wandb_logger,
                      devices=trainer_config.pop('devices', default_devices),
                      accelerator=trainer_config.pop('accelerator', default_accelerator),
                      strategy=trainer_config.pop('strategy', default_strategy),
                      num_sanity_val_steps=trainer_config.pop('num_sanity_val_steps', 0),
                      callbacks=callback_modules,
                      gradient_clip_val=gradient_clip_val,
                      reload_dataloaders_every_n_epochs=reload_dataloaders_interval,
                      check_val_every_n_epoch=val_check_interval,
                      **trainer_config)
    trainer.fit(nf2, data_module, ckpt_path=ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the simulation')
    args, overwrite_args = parser.parse_known_args()

    yaml_config_file = args.config
    config = load_yaml_config(yaml_config_file, overwrite_args)

    run(**config)


if __name__ == '__main__':
    main()
