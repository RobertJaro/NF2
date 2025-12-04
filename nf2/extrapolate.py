import argparse
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from nf2.loader.cartesian import CartesianDataModule
from nf2.loader.spherical import SphericalDataModule
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save
from nf2.train.util import load_yaml_config


def run(base_path, data, work_directory=None, callbacks=[], logging={}, model={}, training={}, loss=[], transforms=[], loss_scaling=[], config=None, reload=False):
    """Run the simulation with the given configuration.

    This function initializes the data loader, the model, the training loop and the logging.
    The simulation is run with the given configuration and the results are stored in the base_path.
    Use the configurations for logging, model and training to overwrite the default settings.

    Args:
        base_path: Path to the directory where the simulation results are stored.
        data: Dictionary with the data loader configuration.
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

    save_path = os.path.join(base_path, 'extrapolation_result.nf2')

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
        checkpoint_reference = f"{logging['entity']}/{logging['project']}/model-{logging['id']}:latest"
        artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
        artifact.download(root=base_path)
        shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
        data['plot_overview'] = False  # skip overview plot for restored model

    # initialize data module
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')
    @rank_zero_only
    def _init_data_module():
        data_module_type = data.pop('type')
        if data_module_type == 'cartesian':
            data_module = CartesianDataModule(**data)
        elif data_module_type == 'spherical':
            data_module = SphericalDataModule(**data)
        else:
            raise NotImplementedError(f'Unknown data loader {data_module_type}')
        torch.save(data_module, data_module_save_path)
    _init_data_module()
    # load data module for all ranks
    data_module = torch.load(data_module_save_path)

    # initialize callbacks
    callback_modules = load_callbacks(callbacks, data_module)

    nf2 = NF2Module(data_module.validation_dataset_mapping, data_module.config,
                    model_kwargs=model, loss_config=loss, transforms=transforms, loss_scaling=loss_scaling)

    config_dict = {'data': data, 'model': model, 'training': training, 'config': config}
    val_check_interval = int(training['validation_interval']) if "validation_interval" in training else None
    val_every_n_epochs = training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training else None
    max_epochs = int(training['epochs']) if 'epochs' in training else 10

    save_callback = LambdaCallback(
        on_validation_end=lambda *_: save(save_path, nf2, data_module, config_dict))
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          every_n_train_steps=val_check_interval,
                                          every_n_epochs=val_every_n_epochs,
                                          save_last=True)

    torch.set_float32_matmul_precision('medium')  # for A100 GPUs
    n_gpus = torch.cuda.device_count()
    callback_modules += [checkpoint_callback, save_callback]

    trainer = Trainer(max_epochs=max_epochs,
                      logger=wandb_logger,
                      devices=n_gpus if n_gpus > 0 else None,
                      accelerator='gpu' if n_gpus >= 1 else None,
                      strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
                      num_sanity_val_steps=0,
                      val_check_interval=val_check_interval,
                      check_val_every_n_epoch=val_every_n_epochs,
                      gradient_clip_val=0.1,
                      callbacks=callback_modules)

    trainer.fit(nf2, data_module, ckpt_path='last')
    save(save_path, nf2, data_module, config_dict)
    # clean up
    data_module.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file for the simulation')
    args, overwrite_args = parser.parse_known_args()

    yaml_config_file = args.config
    config = load_yaml_config(yaml_config_file, overwrite_args)

    run(**config)


if __name__ == '__main__':
    main()
