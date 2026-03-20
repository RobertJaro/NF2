from __future__ import annotations

import glob
import os
import shutil
from copy import deepcopy

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from nf2.core.adapters import resolve_geometry_adapter
import nf2.geometry.cartesian  # noqa: F401
import nf2.geometry.spherical  # noqa: F401
from nf2.train.callback import AdvanceDatamoduleStep
from nf2.train.mapping import load_callbacks
from nf2.train.module import NF2Module, save


def _prepare_run_config(
    base_path,
    data,
    work_directory=None,
    callbacks=None,
    logging=None,
    model=None,
    training=None,
    loss=None,
    transforms=None,
    loss_scaling=None,
    config=None,
):
    callbacks = [] if callbacks is None else deepcopy(callbacks)
    logging = {} if logging is None else deepcopy(logging)
    model = {} if model is None else deepcopy(model)
    training = {} if training is None else deepcopy(training)
    transforms = [] if transforms is None else deepcopy(transforms)
    loss_scaling = [] if loss_scaling is None else deepcopy(loss_scaling)
    data = deepcopy(data)

    if loss is None:
        loss = training.pop("loss_config", [])
    else:
        loss = deepcopy(loss)

    if work_directory is None:
        work_directory = os.path.join(base_path, "work")

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(work_directory, exist_ok=True)
    data["work_directory"] = work_directory

    return {
        "base_path": base_path,
        "work_directory": work_directory,
        "data": data,
        "callbacks": callbacks,
        "logging": logging,
        "model": model,
        "training": training,
        "loss": loss,
        "transforms": transforms,
        "loss_scaling": loss_scaling,
        "config": config,
    }


def _build_logger(base_path, work_directory, logging, model, training, data, loss, transforms, config):
    wandb_logger = WandbLogger(**logging, save_dir=work_directory)
    config_dict = {
        "base_path": base_path,
        "work_directory": work_directory,
        "logging": logging,
        "model": model,
        "training": training,
        "config": config,
        "data": data,
        "loss": loss,
        "transforms": transforms,
    }

    @rank_zero_only
    def _log_hparams(cfg):
        wandb_logger.log_hyperparams(cfg)

    _log_hparams(config_dict)
    return wandb_logger


def _restore_from_wandb(logging, wandb_logger, base_path, data):
    if "id" not in logging:
        return
    if "entity" not in logging or "project" not in logging:
        raise ValueError('"entity" and "project" must be provided to continue from wandb checkpoint')
    checkpoint_reference = f"{logging['entity']}/{logging['project']}/model-{logging['id']}:latest"
    artifact = wandb_logger.use_artifact(checkpoint_reference, artifact_type="model")
    artifact.download(root=base_path)
    shutil.move(os.path.join(base_path, "model.ckpt"), os.path.join(base_path, "last.ckpt"))
    data["plot_overview"] = False


def _save_and_load_datamodule(factory, save_path):
    @rank_zero_only
    def _init_data_module():
        data_module = factory()
        torch.save(data_module, save_path)

    _init_data_module()
    return torch.load(save_path, weights_only=False)


def _build_trainer(training, wandb_logger, callback_modules, *, reload_dataloaders_every_n_epochs=None):
    torch.set_float32_matmul_precision("medium")
    n_gpus = torch.cuda.device_count()
    val_check_interval = int(training["validation_interval"]) if "validation_interval" in training else None
    if reload_dataloaders_every_n_epochs is not None:
        val_every_n_epochs = training.get("check_val_every_n_epoch", 1)
    else:
        val_every_n_epochs = training.get("check_val_every_n_epoch")
    max_epochs = int(training["epochs"]) if "epochs" in training else 10
    if reload_dataloaders_every_n_epochs is not None:
        max_epochs = -1

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "logger": wandb_logger,
        "devices": n_gpus if n_gpus > 0 else None,
        "accelerator": "gpu" if n_gpus >= 1 else None,
        "strategy": DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        "num_sanity_val_steps": 0,
        "callbacks": callback_modules,
        "gradient_clip_val": 0.1,
        "check_val_every_n_epoch": val_every_n_epochs,
    }
    if val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = val_check_interval
    if reload_dataloaders_every_n_epochs is not None:
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = reload_dataloaders_every_n_epochs
    return Trainer(**trainer_kwargs)


def run(
    base_path,
    data,
    work_directory=None,
    callbacks=None,
    logging=None,
    model=None,
    training=None,
    loss=None,
    transforms=None,
    loss_scaling=None,
    config=None,
    reload=False,
):
    del reload
    run_config = _prepare_run_config(
        base_path=base_path,
        data=data,
        work_directory=work_directory,
        callbacks=callbacks,
        logging=logging,
        model=model,
        training=training,
        loss=loss,
        transforms=transforms,
        loss_scaling=loss_scaling,
        config=config,
    )
    adapter = resolve_geometry_adapter(run_config["data"], series=False)
    run_config["data"] = adapter.prepare_data_config(run_config["data"], series=False)

    save_path = os.path.join(base_path, "extrapolation_result.nf2")
    wandb_logger = _build_logger(
        base_path,
        run_config["work_directory"],
        run_config["logging"],
        run_config["model"],
        run_config["training"],
        run_config["data"],
        run_config["loss"],
        run_config["transforms"],
        config,
    )
    _restore_from_wandb(run_config["logging"], wandb_logger, base_path, run_config["data"])

    data_module = _save_and_load_datamodule(
        lambda: adapter.create_data_module(run_config["data"]),
        os.path.join(run_config["work_directory"], "data_module.pkl"),
    )
    callback_modules = load_callbacks(run_config["callbacks"], data_module)
    nf2 = NF2Module(
        data_module.validation_dataset_mapping,
        data_module.config,
        model_kwargs=run_config["model"],
        loss_config=run_config["loss"],
        transforms=run_config["transforms"],
        loss_scaling=run_config["loss_scaling"],
    )

    trainer_config = {
        "data": run_config["data"],
        "model": run_config["model"],
        "training": run_config["training"],
        "config": config,
    }
    val_check_interval = int(run_config["training"]["validation_interval"]) if "validation_interval" in run_config["training"] else None
    val_every_n_epochs = run_config["training"].get("check_val_every_n_epoch")
    save_callback = LambdaCallback(on_validation_end=lambda *_: save(save_path, nf2, data_module, trainer_config))
    checkpoint_callback = ModelCheckpoint(
        dirpath=base_path,
        every_n_train_steps=val_check_interval,
        every_n_epochs=val_every_n_epochs,
        save_last=True,
    )
    callback_modules.extend([checkpoint_callback, save_callback])

    trainer = _build_trainer(run_config["training"], wandb_logger, callback_modules)
    trainer.fit(nf2, data_module, ckpt_path="last")
    save(save_path, nf2, data_module, trainer_config)
    data_module.clear()


def run_series(
    base_path,
    data,
    meta_path,
    work_directory=None,
    callbacks=None,
    logging=None,
    model=None,
    training=None,
    loss=None,
    transforms=None,
    loss_scaling=None,
    config=None,
):
    run_config = _prepare_run_config(
        base_path=base_path,
        data=data,
        work_directory=work_directory,
        callbacks=callbacks,
        logging=logging,
        model=model,
        training=training,
        loss=loss,
        transforms=transforms,
        loss_scaling=loss_scaling,
        config=config,
    )
    adapter = resolve_geometry_adapter(run_config["data"], series=True)
    run_config["data"] = adapter.prepare_data_config(run_config["data"], series=True)

    wandb_logger = _build_logger(
        base_path,
        run_config["work_directory"],
        run_config["logging"],
        run_config["model"],
        run_config["training"],
        run_config["data"],
        run_config["loss"],
        run_config["transforms"],
        config,
    )
    _restore_from_wandb(run_config["logging"], wandb_logger, base_path, run_config["data"])

    ckpts = sorted(glob.glob(os.path.join(base_path, "*.nf2")))
    current_step = len(ckpts)
    ckpt_path = "last" if current_step > 0 else meta_path
    data_module = _save_and_load_datamodule(
        lambda: adapter.create_series_data_module(run_config["data"], current_step=current_step),
        os.path.join(run_config["work_directory"], "data_module.pkl"),
    )
    callback_modules = load_callbacks(run_config["callbacks"], data_module)
    nf2 = NF2Module(
        data_module.validation_dataset_mapping,
        data_module.config,
        model_kwargs=run_config["model"],
        loss_config=run_config["loss"],
        transforms=run_config["transforms"],
        loss_scaling=run_config["loss_scaling"],
    )

    reload_dataloaders_interval = run_config["training"].get("reload_dataloaders_every_n_epochs", 1)
    trainer_config = {
        "data": run_config["data"],
        "model": run_config["model"],
        "training": run_config["training"],
        "config": config,
    }
    save_callback = LambdaCallback(
        on_train_epoch_end=lambda *_: save(
            os.path.join(base_path, data_module.current_id + ".nf2"),
            nf2,
            data_module,
            trainer_config,
        )
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=base_path,
        every_n_epochs=reload_dataloaders_interval,
        save_last=True,
    )
    advance_data_module_callback = AdvanceDatamoduleStep(data_module, reload_dataloaders_interval)
    callback_modules.extend([checkpoint_callback, save_callback, advance_data_module_callback])

    trainer = _build_trainer(
        run_config["training"],
        wandb_logger,
        callback_modules,
        reload_dataloaders_every_n_epochs=reload_dataloaders_interval,
    )
    trainer.fit(nf2, data_module, ckpt_path=ckpt_path)
