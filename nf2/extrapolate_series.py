import argparse
import glob
import os
import shutil

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nf2.loader.fits import FITSSeriesDataModule
from nf2.loader.spherical import SphericalSeriesDataModule
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save
from nf2.train.util import load_yaml_config


def run(base_path, data, meta_path, work_directory=None, logging={}, model={}, training={}, config=None):
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
                   'model': model, 'training': training, 'config': config}
    wandb_logger.experiment.config.update(config_dict, allow_val_change=True)

    # restore model checkpoint from wandb
    if 'id' in logging:
        assert 'entity' in logging and 'project' in logging, '"entity" and "project" must be provided to continue from wandb checkpoint'
        checkpoint_reference = f"{logging['entity']}/{logging['project']}/model-{logging['id']}:latest"
        artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
        artifact.download(root=base_path)
        shutil.move(os.path.join(base_path, 'model.ckpt'), os.path.join(base_path, 'last.ckpt'))
        data['plot_overview'] = False  # skip overview plot for restored model

    fits_paths, error_paths = _load_paths(data['data_path'])

    # reload model training
    ckpts = sorted(glob.glob(os.path.join(base_path, '*.nf2')))
    ckpt_path = 'last' if len(ckpts) > 0 else meta_path  # reload last savepoint
    fits_paths = fits_paths[len(ckpts):]  # select remaining extrapolations
    error_paths = error_paths[len(ckpts):] if error_paths is not None else None

    # initialize data module
    if data["type"] == 'sharp':
        data_module = FITSSeriesDataModule(fits_paths, error_paths=error_paths, **data)
    elif data["type"] == 'spherical':
        data_module = SphericalSeriesDataModule(fits_paths, **data)
    else:
        raise NotImplementedError(f'Unknown data loader {data["type"]}')

    callbacks = load_callbacks(data_module)

    nf2 = NF2Module(data_module.validation_dataset_mapping, data_module.config, model_kwargs=model, **training)

    reload_dataloaders_interval = training[
        'reload_dataloaders_every_n_epochs'] if 'reload_dataloaders_every_n_epochs' in training else 1

    # callback
    config_dict = {'data': data, 'model': model, 'training': training, 'config': config}
    save_callback = LambdaCallback(
        on_train_epoch_end=lambda *args:
        save(os.path.join(base_path, data_module.current_id + '.nf2'),
             nf2, data_module, config_dict))

    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          every_n_epochs=reload_dataloaders_interval,
                                          save_last=True)

    # general training parameters
    torch.set_float32_matmul_precision('medium')  # for A100 GPUs
    n_gpus = torch.cuda.device_count()

    val_check_interval = training['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training else 1
    trainer = Trainer(max_epochs=-1,
                      logger=wandb_logger,
                      devices=n_gpus,
                      accelerator='gpu' if n_gpus >= 1 else None,
                      strategy='dp' if n_gpus > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=0, callbacks=[save_callback, checkpoint_callback, *callbacks],
                      gradient_clip_val=0.1, reload_dataloaders_every_n_epochs=reload_dataloaders_interval,
                      check_val_every_n_epoch=val_check_interval)
    trainer.fit(nf2, data_module, ckpt_path=ckpt_path)


def _load_paths(data_path):
    if isinstance(data_path, list):
        results = [_load_paths(d) for d in data_path]
        fits_paths = [f for r in results for f in r[0]]
        error_paths = [f for r in results for f in r[1]] if all([r[1] is not None for r in results]) else None
    elif isinstance(data_path, str):
        p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
        t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
        r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
        err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
        err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
        err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

        assert len(p_files) == len(t_files) == len(r_files), f'Number of files in data path {data_path} does not match'
        fits_paths = list(zip(p_files, t_files, r_files))
        fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_paths]

        if len(err_p_files) > 0 or len(err_t_files) > 0 or len(err_r_files) > 0:
            assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(
                err_r_files), \
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
            assert len(p_files) == len(err_p_files) == len(t_files) == len(err_t_files) == len(r_files) == len(
                err_r_files), \
                f'Number of files in data path {data_path} does not match'
            fits_paths = list(zip(p_files, t_files, r_files))
            fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2], } for d in fits_paths]
            error_paths = list(zip(err_p_files, err_t_files, err_r_files))
            error_paths = [{'Bp_err': d[0], 'Bt_err': d[1], 'Br_err': d[2], } for d in error_paths]
        else:
            assert len(p_files) == len(t_files) == len(r_files), \
                f'Number of files in data path {data_path} does not match'
            fits_paths = list(zip(p_files, t_files, r_files))
            fits_paths = [{'Bp': d[0], 'Bt': d[1], 'Br': d[2]} for d in fits_paths]
            error_paths = None
    else:
        raise NotImplementedError(f'Unknown data path type {type(data_path)}')
    return fits_paths, error_paths


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
