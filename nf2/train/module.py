import copy
import logging

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nf2.train.loss import loss_module_mapping
from nf2.train.model import BModel, VectorPotentialModel, MagnetoStaticModel, MagnetoStaticModelV2
from nf2.train.transform import HeightTransformModel, AzimuthTransformModel


class NF2Module(LightningModule):

    def __init__(self, validation_mapping, data_config, model_kwargs=None, loss_config=None,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 transforms=[], meta_path=None, **kwargs):
        """
        The main module for training the neural field model.

        Args:
            validation_mapping (dict): Dictionary mapping validation dataset indices to data loader indices.
            dim (int): Dimension of the neural field model.
            loss_config (list): List of dictionaries containing the loss type and lambda value.
            lr_params (dict): Dictionary containing the start, end and iterations for the learning rate scheduler.
            meta_path (str): Path to a meta state file.
            use_positional_encoding (bool): Whether to use positional encoding.
            model (str): Model type, one of ['b', 'vector_potential', 'flux'].
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        model_kwargs = model_kwargs if model_kwargs is not None else {'type': 'b', 'dim': 256}
        # init model
        model_kwargs = copy.deepcopy(model_kwargs)
        model_type = model_kwargs.pop('type')
        if model_type == 'b':
            model = BModel(**model_kwargs)
        elif model_type == 'vector_potential':
            model = VectorPotentialModel(**model_kwargs)
        elif model_type == 'magneto_static':
            model = MagnetoStaticModel(**model_kwargs)
        elif model_type == 'magneto_static_v2':
            model = MagnetoStaticModelV2(**model_kwargs)
        else:
            raise ValueError(f"Invalid model: {model_type}, must be in ['b', 'vector_potential', 'flux']")

        # init coordinate mapping model
        transform_modules = self.load_transfrom_config(transforms)
        self.transform_modules = nn.ModuleList(transform_modules)

        # load meta state
        if meta_path:
            state_dict = torch.load(meta_path)['model'].state_dict() \
                if meta_path.endswith('nf2') else torch.load(meta_path)['m']
            model.load_state_dict(state_dict)
            logging.info('Loaded meta state: %s' % meta_path)

        # init loss modules
        if loss_config is None:
            logging.info('Using default loss configuration')
            loss_config = [
                {"type": "boundary", "lambda": {"start": 1e3, "end": 1, "iterations": 1e5},
                 'ds_id': ['boundary_01', 'potential']},
                {"type": "force_free", "lambda": 1e-1},
                {"type": "divergence", "lambda": 1e-1},
            ]

        # init lambdas and loss modules
        loss_modules, lambdas, scheduled_lambdas = self.load_loss_config(data_config, loss_config)
        self.loss_modules = nn.ModuleDict(loss_modules)
        self.scheduled_lambdas = nn.ParameterDict(scheduled_lambdas)
        self.lambdas = nn.ParameterDict(lambdas)

        self.model = model
        self.validation_mapping = validation_mapping
        self.lr_params = lr_params

        #
        self.validation_outputs = {}

    def load_transfrom_config(self, transforms):
        transform_modules = []
        for transform_config in transforms:
            transform_config = copy.deepcopy(transform_config)

            transform_type = transform_config.pop('type')
            ds_ids = transform_config.pop('ds_id')
            if transform_type == 'height':
                transform_module = HeightTransformModel(**transform_config, ds_id=ds_ids)
            elif transform_type == 'azimuth':
                transform_module = AzimuthTransformModel(**transform_config, ds_id=ds_ids)
            else:
                raise ValueError(f"Invalid coordinate mapping model: {transform_type}")
            transform_modules.append(transform_module)
        return transform_modules

    def load_loss_config(self, data_config, loss_config):
        scheduled_lambdas = {}
        lambdas = {}
        loss_modules = {}
        loss_config = copy.deepcopy(loss_config)
        for config in loss_config:
            k = config.pop('type')
            l = config.pop('lambda')
            # update dataset id
            ds_id = config.pop('ds_id', 'all')
            config['ds_id'] = ds_id
            # update name
            name = config.pop('name', None)
            if name is None:
                ds_str = '_'.join(ds_id) if isinstance(ds_id, list) else ds_id
                name = f'{ds_str}--{k}' if ds_str != 'all' else k
            config['name'] = name
            if isinstance(l, dict):
                value = torch.tensor(l['start'], dtype=torch.float32)
                lt = 'exponential' if 'type' not in l else l['type']
                if lt == 'exponential':
                    gamma = torch.tensor((l['end'] / l['start']) ** (1 / l['iterations']), dtype=torch.float32)
                elif lt == 'linear':
                    gamma = torch.tensor((l['end'] - l['start']) / l['iterations'], dtype=torch.float32)
                else:
                    raise ValueError(f"Invalid lambda type: {lt}, must be in ['exponential', 'linear']")
                end = torch.tensor(l['end'], dtype=torch.float32)
                scheduled_lambdas[name] = {'gamma': gamma, 'end': end, 'type': lt}
            else:
                value = torch.tensor(l, dtype=torch.float32)
            assert name not in lambdas, f"Duplicate name for loss: {name}"
            lambdas[name] = value
            # additional kwargs
            loss_module = {name: loss_module_mapping[k](**config, **data_config)}
            loss_modules.update(loss_module)
        return loss_modules, lambdas, scheduled_lambdas

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        parameters += list(self.transform_modules.parameters())
        parameters += list(self.loss_modules.parameters())
        if isinstance(self.lr_params, dict):
            lr_start = self.lr_params['start']
            lr_end = self.lr_params['end']
            iterations = self.lr_params['iterations']
        elif isinstance(self.lr_params, (float, int)):
            lr_start = self.lr_params
            lr_end = self.lr_params
            iterations = 1
            self.lr_params = {'start': lr_start, 'end': lr_end, 'iterations': iterations}
        else:
            raise ValueError(f"Invalid lr_params: {self.lr_params}, must be dict or float/int")
        optimizer = torch.optim.Adam(parameters, lr=lr_start)
        scheduler = ExponentialLR(optimizer, gamma=(lr_end / lr_start) ** (1 / iterations))

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        loader_keys = list(batch.keys())

        # set requires grad for coords
        for k in loader_keys:
            batch[k]['coords'].requires_grad = True

        # transform batch
        self.apply_transforms(batch)

        # concatenate all points
        coords = torch.cat([batch[k]['coords'] for k in loader_keys], 0)

        # forward step
        model_out = self.model(coords)
        model_out_keys = model_out.keys()

        # split back into dataloaders
        result_mapping = {}
        idx = 0
        for k in loader_keys:
            n_coords = batch[k]['coords'].shape[0]
            result_mapping[k] = {mk: model_out[mk][idx:idx + n_coords] for mk in model_out_keys}
            idx += n_coords

        state_dict = {k: {**result_mapping[k], **batch[k]} for k in loader_keys}

        # global state
        state_dict['all'] = {**model_out, 'coords': coords}

        loss_dict = {}
        for name, loss_module in self.loss_modules.items():
            try:
                ds_id = loss_module.ds_id
                if isinstance(ds_id, list):
                    states = [state_dict[i] for i in ds_id]
                    state_keys = states[0].keys()
                    state = {k: torch.cat([s[k] for s in states]) for k in state_keys}
                else:
                    state = state_dict[ds_id]
                loss = {name: loss_module(**state)}
                loss_dict.update(loss)
            except Exception as e:
                logging.error(f"Error in loss module: {name}")
                raise e
        total_loss = sum([self.lambdas[k] * loss_dict[k] for k in loss_dict.keys()])

        loss_dict['loss'] = total_loss
        return loss_dict

    def apply_transforms(self, batch):
        loader_keys = list(batch.keys())
        for transform_module in self.transform_modules:
            ds_ids = transform_module.ds_id
            tensor_ids = transform_module.tensor_ids
            # concatenate all tensors
            transform_ds_ids = [k for k in loader_keys if k in ds_ids]
            transform_batch = {tensor_id: torch.cat([batch[k][tensor_id] for k in transform_ds_ids], 0) for tensor_id in
                               tensor_ids}
            batch_lengths = [batch[k][tensor_ids[0]].shape[0] for k in transform_ds_ids]
            transformed_batch = transform_module(transform_batch)

            # update batch
            for k in transformed_batch.keys():
                value = transformed_batch[k]

                number_idx = 0
                for ds_id, batch_length in zip(transform_ds_ids, batch_lengths):
                    batch[ds_id][k] = value[number_idx:number_idx + batch_length]
                    number_idx += batch_length

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update lambda parameters and log
        for k in self.scheduled_lambdas.keys():
            param = self.lambdas[k]
            gamma = self.scheduled_lambdas[k]['gamma']
            if self.scheduled_lambdas[k]['type'] == 'exponential':
                if (gamma < 1 and param > self.scheduled_lambdas[k]['end']) or \
                        (gamma > 1 and param < self.scheduled_lambdas[k]['end']):
                    new_lambda = param * gamma
                    param.copy_(new_lambda)
            if self.scheduled_lambdas[k]['type'] == 'linear':
                if (gamma > 0 and param < self.scheduled_lambdas[k]['end']) or \
                        (gamma < 0 and param > self.scheduled_lambdas[k]['end']):
                    new_lambda = param + gamma
                    param.copy_(new_lambda)
            self.log('lambda_' + k, self.lambdas[k])
        # update learning rate
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        assert not torch.isnan(outputs['loss'].mean()), f"Loss is NaN. Check input data and run configuration."
        # log results to WANDB
        self.log("train", {k: v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx):
        coords = batch['coords']
        coords.requires_grad = True

        for transform_module in self.transform_modules:
            ds_ids = transform_module.ds_id
            if self.validation_mapping[dataloader_idx] in ds_ids:
                transform_out = transform_module(batch)
                for k, v in transform_out.items():
                    batch[k] = v

        coords = batch['coords']
        model_out = self.model(coords)

        jac_matrix = model_out['jac_matrix']
        dBx_dx = jac_matrix[:, 0, 0]
        dBy_dx = jac_matrix[:, 1, 0]
        dBz_dx = jac_matrix[:, 2, 0]
        dBx_dy = jac_matrix[:, 0, 1]
        dBy_dy = jac_matrix[:, 1, 1]
        dBz_dy = jac_matrix[:, 2, 1]
        dBx_dz = jac_matrix[:, 0, 2]
        dBy_dz = jac_matrix[:, 1, 2]
        dBz_dz = jac_matrix[:, 2, 2]
        #
        rot_x = dBz_dy - dBy_dz
        rot_y = dBx_dz - dBz_dx
        rot_z = dBy_dx - dBx_dy
        #
        j = torch.stack([rot_x, rot_y, rot_z], -1)
        div = torch.abs(dBx_dx + dBy_dy + dBz_dz)

        model_out['j'] = j
        model_out['div'] = div

        return {k: v.detach() for k, v in {**batch, **model_out}.items()}

    def validation_epoch_end(self, outputs_list):
        self.validation_outputs = {}  # reset validation outputs
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

        for i, outputs in enumerate(outputs_list):
            out_keys = outputs[0].keys()
            outputs = {k: torch.cat([o[k] for o in outputs]) for k in out_keys}
            self.validation_outputs[self.validation_mapping[i]] = outputs

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        # keep new lambdas
        for k, v in self.lambdas.items():
            if f"lambdas.{k}" not in state_dict:
                print(f'Add lambda {k}: {v.data}')
                state_dict[f'lambdas.{k}'] = v
                continue
            checkpoint_v = state_dict[f"lambdas.{k}"]
            if k in self.scheduled_lambdas or checkpoint_v == v:  # skip scheduled lambdas or same values
                continue
            print(f'Update lambda {k}: {checkpoint_v} --> {v.data}')
            state_dict[f'lambdas.{k}'] = v
        # remove old lambdas
        remove_keys = []
        for k, v in state_dict.items():
            if 'lambdas' in k and k.split('.')[1] not in self.lambdas.keys():
                print(f'Remove lambda: {k}')
                remove_keys.append(k)
        [state_dict.pop(k) for k in remove_keys]
        # remove old scheduled lambdas
        remove_keys = []
        for k, v in state_dict.items():
            if 'scheduled_lambdas' in k and k.split('.')[1] not in self.scheduled_lambdas.keys():
                print(f'Remove scheduled lambda: {k}')
                remove_keys.append(k)
        [state_dict.pop(k) for k in remove_keys]

        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs


def save(save_path, nf2, data_module, config):
    save_state = {'model': nf2.model,
                  'config': config,
                  'data': data_module.config,
                  'transforms': nf2.transform_modules, }
    torch.save(save_state, save_path)
