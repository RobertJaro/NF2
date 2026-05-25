import argparse
import glob
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from nf2.loader.cartesian import CartesianSeriesDataModule
from nf2.loader.spherical import SphericalSeriesDataModule
from nf2.train.callback import AdvanceDatamoduleStep
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save
from nf2.train.util import load_yaml_config


def run(base_path, data, meta_path, work_directory=None, callbacks=[], logging={}, model={}, training={}, loss=[],
        transforms=[], loss_scaling=[], config=None):
    """Run the simulation with the given configuration.

    This function initializes the data loader, the model, the training loop and the logging.
    The simulation is run with the given configuration and the results are stored in the base_path.
    Use the configurations for logging, model and training to overwrite the default settings.

    Args:
        base_path: Path to the directory where the simulation results are stored.
        data: Dictionary with the data loader configuration.
        meta_path: Path to the initial model checkpoint (run extrapolation first).
        work_directory: Path to the directory where the data is stored. If None, the base_path is used.
        logging: Dictionary with the logging configuration.
        model: Dictionary with the model configuration.
        training: Dictionary with the training configuration.
        config: Dictionary with the configuration for the simulation.
    """
    os.makedirs(base_path, exist_ok=True)

    if work_directory is None:
        work_directory = os.path.join(base_path, 'work')
    os.makedirs(work_directory, exist_ok=True)
    data['work_directory'] = work_directory  # set work directory for data loaders

    # init logging
    wandb_logger = WandbLogger(**logging, save_dir=work_directory)
    config_dict = {'base_path': base_path, 'work_directory': work_directory, 'logging': logging,
                   'model': model, 'training': training, 'config': config, 'data': data,
                   'loss': loss, 'transforms': transforms}

    @rank_zero_only
    def _log_hparams(cfg):
        wandb_logger.log_hyperparams(cfg)

    _log_hparams(config_dict)

    # restore model checkpoint from wandb
    if 'id' in logging:
        assert 'entity' in logging and 'project' in logging, '"entity" and "project" must be provided to continue from wandb checkpoint'
        checkpoint_reference = f"{logging['entity']}/{logging['project']}/model-{logging['id']}:latest"
        artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
        artifact.download(root=base_path)
        shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
        data['plot_overview'] = False  # skip overview plot for restored model

    # reload model training
    ckpts = sorted(glob.glob(os.path.join(base_path, '*.nf2')))
    current_step = len(ckpts)
    last_ckpt_path = os.path.join(base_path, 'last.ckpt')
    ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    meta_state_path = None if ckpt_path is not None else meta_path

    # initialize data module
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')

    @rank_zero_only
    def _init_data_module():
        assert 'type' in data, 'Data module type must be specified in the configuration'
        data_module_type = data.pop('type')
        if data_module_type == 'cartesian':
            data_module = CartesianSeriesDataModule(current_step=current_step, **data)
        elif data_module_type == 'spherical':
            data_module = SphericalSeriesDataModule(current_step=current_step, **data)
        else:
            raise NotImplementedError(f'Unknown data loader {data_module_type}')
        torch.save(data_module, data_module_save_path)

    _init_data_module()
    # load data module for all ranks
    data_module = torch.load(data_module_save_path, weights_only=False)

    callback_modules = load_callbacks(callbacks, data_module)

    nf2 = NF2Module(data_module.validation_dataset_mapping, data_module.config,
                    model_kwargs=model, loss_config=loss, transforms=transforms, loss_scaling=loss_scaling,
                    meta_path=meta_state_path)

    reload_dataloaders_interval = training[
        'reload_dataloaders_every_n_epochs'] if 'reload_dataloaders_every_n_epochs' in training else 1
    max_epochs = data_module.total_steps * reload_dataloaders_interval if ckpt_path is not None else \
        (data_module.total_steps - data_module.step) * reload_dataloaders_interval

    # callback
    config_dict = {'data': data, 'model': model, 'training': training, 'config': config}
    save_callback = LambdaCallback(
        on_train_epoch_end=lambda *args:
        save(os.path.join(base_path, data_module.current_id + '.nf2'),
             nf2, data_module, config_dict))

    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          every_n_epochs=reload_dataloaders_interval,
                                          save_last=True)

    advance_data_module_callback = AdvanceDatamoduleStep(data_module, reload_dataloaders_interval)

    # general training parameters
    torch.set_float32_matmul_precision('medium')  # for A100 GPUs
    n_gpus = torch.cuda.device_count()
    callback_modules += [checkpoint_callback, save_callback, advance_data_module_callback]

    val_check_interval = training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training else 1
    trainer = Trainer(max_epochs=max_epochs,
                      logger=wandb_logger,
                      devices=n_gpus if n_gpus > 0 else None,
                      accelerator='gpu' if n_gpus >= 1 else None,
                      strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
                      num_sanity_val_steps=0,
                      callbacks=callback_modules,
                      gradient_clip_val=0.1, reload_dataloaders_every_n_epochs=reload_dataloaders_interval,
                      check_val_every_n_epoch=val_check_interval)
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
