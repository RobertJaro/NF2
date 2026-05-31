import copy
from importlib.metadata import PackageNotFoundError, version
import logging

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nf2.train.loss import loss_module_mapping
from nf2.train.loss_scaling import ExponentialLossScalingModule, PotentialFitLossScalingModule, \
    BHeightLossScalingModule, RadialLossScalingModule
from nf2.train.model import BModel, VectorPotentialModel
from nf2.train.transform import HeightRangeTransformModel, AzimuthTransformModel, OpticalDepthTransformModel, \
    HeightTransformModel


class NF2Module(LightningModule):

    def __init__(self, validation_mapping, data_config, model_kwargs=None, loss_config=None,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 transforms=[], loss_scaling=[], meta_path=None, **kwargs):
        """
        The main module for training the neural field model.
    
        Args:
            validation_mapping (dict): Dictionary mapping validation dataset indices to data loader indices.
            data_config (dict): Configuration dictionary containing data parameters like coordinate ranges,
                              dataset-specific parameters, and data preprocessing settings.
            model_kwargs (dict, optional): Model configuration dictionary containing:
                - type (str): Model type, one of ['b', 'vector_potential']
                - dim (int): Hidden dimension size of the neural network
                Additional model-specific parameters
            loss_config (list, optional): List of dictionaries containing loss configurations:
                - type (str): Loss type (e.g., 'boundary', 'force_free', 'divergence')
                - weight (float or dict): Loss weight value or schedule configuration
                - ds_id (str or list): Dataset ID(s) to apply the loss to
                - Additional loss-specific parameters
            lr_params (dict): Learning rate scheduler configuration containing:
                - start (float): Initial learning rate
                - end (float): Final learning rate
                - iterations (int): Number of iterations for the schedule
            transforms (list): List of coordinate transform configurations
            meta_path (str, optional): Path to a pretrained model state file
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        Mm_per_ds = data_config['Mm_per_ds']
        Gauss_per_dB = data_config['Gauss_per_dB']
        self.Mm_per_ds = Mm_per_ds

        if 'coord_range' in data_config:
            coord_ranges = np.array(data_config['coord_range'])
            coord_range = np.stack([coord_ranges[:, :, 0].min(0),
                                    coord_ranges[:, :, 1].max(0)], -1)
        else:
            coord_range = None
        model_kwargs = model_kwargs if model_kwargs is not None else {'type': 'b', 'dim': 256}
        model_kwargs['coord_range'] = coord_range
        model_kwargs['ds_per_pixel'] = max(data_config['ds_per_pixel']) if 'ds_per_pixel' in data_config else 1
        # init model
        model_kwargs = copy.deepcopy(model_kwargs)
        model_type = model_kwargs.pop('type')
        if model_type == 'b':
            model = BModel(**model_kwargs)
        elif model_type == 'vector_potential':
            model = VectorPotentialModel(**model_kwargs)
        else:
            valid_options = ['b', 'vector_potential']
            raise ValueError(f"Invalid model: {model_type}, must be in {valid_options}")

        # init coordinate mapping model
        transform_modules = self.load_transfrom_config(transforms)
        self.transform_modules = nn.ModuleList(transform_modules)

        # load meta state
        if meta_path:
            checkpoint = torch.load(meta_path, map_location='cpu', weights_only=False)
            if meta_path.endswith('nf2'):
                state_dict = checkpoint['model'].state_dict()
            elif 'm' in checkpoint:
                state_dict = checkpoint['m']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()
                              if k.startswith('model.')}
            else:
                raise KeyError(f"Unsupported meta checkpoint format: {meta_path}")
            model.load_state_dict(state_dict)
            logging.info('Loaded meta state: %s' % meta_path)

        # init loss modules
        if loss_config is None:
            logging.info('Using default loss configuration')
            loss_config = [
                {"type": "boundary", "weight": {"start": 1e3, "end": 1, "iterations": 1e5},
                 'ds_id': ['boundary_01', 'potential']},
                {"type": "force_free", "weight": 1e-1},
                {"type": "divergence", "weight": 1e-1},
            ]
        # init loss weights and loss modules
        loss_modules, weights, scheduled_weights = self.load_loss_config(data_config, loss_config)
        self.loss_modules = nn.ModuleDict(loss_modules)
        self.scheduled_weights = scheduled_weights
        self.weights = nn.ParameterDict(weights)

        self.model = model
        self.validation_mapping = validation_mapping
        self.lr_params = lr_params

        loss_scaling_modules = {}
        for scaling_config in loss_scaling:
            scaling_config = copy.deepcopy(scaling_config)
            scaling_type = scaling_config.pop('type')
            if scaling_type == 'exponential':
                name = 'exponential_scaling' if 'name' not in scaling_config else scaling_config.pop('name')
                loss_scaling_modules[name] = ExponentialLossScalingModule(**scaling_config, name=name)
            elif scaling_type == 'potential_fit':
                name = 'potential_fit_scaling' if 'name' not in scaling_config else scaling_config.pop('name')
                loss_scaling_modules[name] = PotentialFitLossScalingModule(**scaling_config, name=name,
                                                                           Mm_per_ds=Mm_per_ds, Gauss_per_dB=Gauss_per_dB)
            elif scaling_type == 'b_height':
                name = 'B_height_scaling' if 'name' not in scaling_config else scaling_config.pop('name')
                loss_scaling_modules[name] = BHeightLossScalingModule(**scaling_config, name=name)
            elif scaling_type == 'radial':
                name = 'radial_scaling' if 'name' not in scaling_config else scaling_config.pop('name')
                loss_scaling_modules[name] = RadialLossScalingModule(**scaling_config, name=name, Mm_per_ds=Mm_per_ds)
            else:
                raise ValueError(f"Invalid loss scaling type: {scaling_type}")
        self.loss_scaling_modules = nn.ModuleDict(loss_scaling_modules)
        #
        for module in self.loss_scaling_modules.values():
            for loss_id in module.loss_ids:
                assert loss_id in self.loss_modules.keys(), f"Loss id {loss_id} in loss scaling module " \
                                                            f"not found in loss modules. Available losses: {list(self.loss_modules.keys())}"
        self.validation_outputs = {}
        self.validation_batches = {}

    def load_transfrom_config(self, transforms):
        transform_modules = []
        for transform_config in transforms:
            transform_config = copy.deepcopy(transform_config)

            transform_type = transform_config.pop('type')
            ds_ids = transform_config.pop('ds_id')
            if transform_type == 'height_range':
                transform_module = HeightRangeTransformModel(**transform_config, ds_id=ds_ids)
            elif transform_type == 'height':
                transform_module = HeightTransformModel(**transform_config, ds_id=ds_ids, Mm_per_ds=self.Mm_per_ds)
            elif transform_type == 'optical_depth':
                transform_module = OpticalDepthTransformModel(**transform_config, ds_id=ds_ids,
                                                              Mm_per_ds=self.Mm_per_ds)
            elif transform_type == 'azimuth':
                transform_module = AzimuthTransformModel(**transform_config, ds_id=ds_ids)
            else:
                raise ValueError(f"Invalid coordinate mapping model: {transform_type}")
            transform_modules.append(transform_module)
        return transform_modules

    def load_loss_config(self, data_config, loss_config):
        scheduled_weights = {}
        weights = {}
        loss_modules = {}
        loss_config = copy.deepcopy(loss_config)
        for config in loss_config:
            k = config.pop('type')
            if 'lambda' in config:
                raise ValueError("Loss key 'lambda' was removed in v0.4. Use 'weight'.")
            l = config.pop('weight')
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
                    gamma = (l['end'] / l['start']) ** (1 / l['iterations'])
                    end = l['end']
                    scheduled_weights[name] = {'gamma': gamma, 'end': end, 'type': lt}
                elif lt == 'linear':
                    gamma = (l['end'] - l['start']) / l['iterations']
                    end = l['end']
                    scheduled_weights[name] = {'gamma': gamma, 'end': end, 'type': lt}
                elif lt == 'step':
                    steps = l['steps']
                    end = l['end']
                    scheduled_weights[name] = {'steps': steps, 'end': end, 'type': lt}
                else:
                    raise ValueError(f"Invalid weight schedule type: {lt}, must be in ['exponential', 'linear', 'step']")
            else:
                value = torch.tensor(l, dtype=torch.float32)
            assert name not in weights, f"Duplicate name for loss: {name}"
            weights[name] = value
            # additional kwargs
            if k not in loss_module_mapping:
                valid_options = ', '.join(sorted(loss_module_mapping))
                raise ValueError(f"Invalid loss type: {k}. Valid options are: {valid_options}")
            loss_module = {name: loss_module_mapping[k](**config, **data_config)}
            loss_modules.update(loss_module)
        return loss_modules, weights, scheduled_weights

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        parameters += list(self.transform_modules.parameters())
        parameters += list(self.loss_modules.parameters())
        parameters += list(self.loss_scaling_modules.parameters())
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

    @staticmethod
    def _collate_states(states):
        state_keys = set.intersection(*(set(s.keys()) for s in states))
        state_keys.discard('requires_jacobian')
        state = {k: torch.cat([s[k] for s in states]) for k in state_keys}

        if 'b_err' not in state and any('b_err' in s for s in states):
            state['b_err'] = torch.cat([
                s['b_err'] if 'b_err' in s else torch.zeros_like(s['b_true'])
                for s in states
            ])

        return state

    @staticmethod
    def _requires_jacobian(state):
        requires_jacobian = state.get('requires_jacobian', True)
        if torch.is_tensor(requires_jacobian):
            return bool(requires_jacobian.flatten()[0].item())
        return bool(requires_jacobian)

    def _forward_dataset_group(self, batch, loader_keys, compute_jacobian):
        if not loader_keys:
            return {}, None, {}

        coords = torch.cat([batch[k]['coords'] for k in loader_keys], 0)
        model_out = self.model(coords, compute_jacobian=compute_jacobian)
        model_out_keys = model_out.keys()

        result_mapping = {}
        idx = 0
        for k in loader_keys:
            n_coords = batch[k]['coords'].shape[0]
            result_mapping[k] = {mk: model_out[mk][idx:idx + n_coords] for mk in model_out_keys}
            idx += n_coords

        return model_out, coords, result_mapping

    def training_step(self, batch, batch_nb):
        loader_keys = list(batch.keys())

        # set requires grad for coords
        for k in loader_keys:
            batch[k]['coords'].requires_grad = True

        # transform batch
        self.apply_transforms(batch)

        jacobian_loader_keys = [k for k in loader_keys if self._requires_jacobian(batch[k])]
        no_jacobian_loader_keys = [k for k in loader_keys if k not in jacobian_loader_keys]

        # forward step
        jacobian_model_out, jacobian_coords, jacobian_result_mapping = self._forward_dataset_group(
            batch, jacobian_loader_keys, compute_jacobian=True)
        _, _, no_jacobian_result_mapping = self._forward_dataset_group(
            batch, no_jacobian_loader_keys, compute_jacobian=False)

        result_mapping = {**jacobian_result_mapping, **no_jacobian_result_mapping}

        state_dict = {k: {**result_mapping[k], **batch[k]} for k in loader_keys}

        # global state for physics losses. Datasets that opt out of gradients do not provide jac_matrix.
        if jacobian_loader_keys:
            state_dict['all'] = {**jacobian_model_out, 'coords': jacobian_coords}
        else:
            state_dict['all'] = self._collate_states([state_dict[k] for k in loader_keys])

        loss_dict = {}
        for name, loss_module in self.loss_modules.items():
            try:
                ds_id = loss_module.ds_id
                if isinstance(ds_id, list):
                    states = [state_dict[i] for i in ds_id]
                    state = self._collate_states(states)
                else:
                    state = state_dict[ds_id]
                loss = loss_module(**state)
                for scaling_module in self.loss_scaling_modules.values():
                    if name in scaling_module.loss_ids:
                        loss = scaling_module(loss, state)
                loss_dict.update({name: loss.mean()})
            except Exception as e:
                logging.error(f"Error in loss module: {name}")
                raise e
        total_loss = sum([self.weights[k] * loss_dict[k] for k in loss_dict.keys()])

        return {**{k: v.detach() for k, v in loss_dict.items()}, 'loss': total_loss}

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
        # update scheduled loss weights and log
        for k in self.scheduled_weights.keys():
            param = self.weights[k]
            schedule_type = self.scheduled_weights[k]['type']
            if schedule_type == 'exponential':
                gamma = self.scheduled_weights[k]['gamma']
                if (gamma < 1 and param > self.scheduled_weights[k]['end']) or \
                        (gamma > 1 and param < self.scheduled_weights[k]['end']):
                    new_weight = param * gamma
                    param.copy_(new_weight)
            if schedule_type == 'linear':
                gamma = self.scheduled_weights[k]['gamma']
                if (gamma > 0 and param < self.scheduled_weights[k]['end']) or \
                        (gamma < 0 and param > self.scheduled_weights[k]['end']):
                    new_weight = param + gamma
                    param.copy_(new_weight)
            if schedule_type == 'step':
                steps = self.scheduled_weights[k]['steps']
                if self.global_step > steps:
                    new_weight = self.scheduled_weights[k]['end']
                    param.copy_(new_weight)
            self.log('weight_' + k, self.weights[k])
        # update learning rate
        scheduler = self.lr_schedulers()
        if scheduler.get_last_lr()[0] > self.lr_params['end']:
            scheduler.step()
        self.log('Learning Rate', scheduler.get_last_lr()[0])

        assert not torch.isnan(outputs['loss'].mean()), f"Loss is NaN. Check input data and run configuration."
        # log results to WANDB
        self.log_dict({f"train.{k}": v.mean() for k, v in outputs.items() if k != 'loss'})
        self.log('train.loss', outputs['loss'].mean(), prog_bar=True)

    def on_validation_epoch_start(self):
        self.validation_batches = {}
        self.validation_outputs = {}

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx=0):
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

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Only rank-0 needs to keep outputs if you're running val on rank-0 only.
        if outputs is not None:
            # ensure CPU to keep GPU mem low
            cpu_out = {k: v.detach().cpu() for k, v in outputs.items()}
            if dataloader_idx not in self.validation_batches:
                self.validation_batches[dataloader_idx] = []
            self.validation_batches[dataloader_idx].append(cpu_out)

    def on_validation_epoch_end(self):
        outputs_list = self.validation_batches
        if not outputs_list or len(outputs_list) == 0:
            return

        if dist.is_initialized(): # gather from all ranks
            rank = dist.get_rank()
            world = dist.get_world_size()
            if rank == 0:
                obj_gather_list = [None] * world
                dist.gather_object(self.validation_batches, obj_gather_list, dst=0)
                # Merge dicts from all ranks
                merged_outputs = {}
                for rank_dict in obj_gather_list:
                    for dataloader_idx, batch_list in rank_dict.items():
                        if dataloader_idx not in merged_outputs:
                            merged_outputs[dataloader_idx] = []
                        merged_outputs[dataloader_idx].extend(batch_list)
                outputs_list = merged_outputs
            else:
                dist.gather_object(self.validation_batches, None, dst=0)
                return

        for dataloader_idx, outputs in outputs_list.items():
            # ---- reorder the list itself ----
            # get a single scalar lin_idx for each batch element
            # (use mean or first value if it's a vector)
            idxs = []
            for i, out in enumerate(outputs):
                lin_idx = out.pop('dataset_idx') # for sorting; discard for later steps
                if any([lin_idx == li for li, _ in idxs]):
                    continue # duplicated by DDP
                if lin_idx.ndim > 0:
                    lin_idx = lin_idx.view(-1)[0]  # take first sample in that batch
                idxs.append((int(lin_idx), i))
            # sort by the scalar lin_idx
            outputs = [outputs[i] for _, i in sorted(idxs, key=lambda x: x[0])]
            cube_shape = outputs[0].get('cube_shape')
            # ---- concatenate outputs ----
            out_keys = outputs[0].keys()
            outputs = {k: torch.cat([o[k] for o in outputs]) for k in out_keys}
            if cube_shape is not None:
                outputs['cube_shape'] = cube_shape.reshape(-1)
            self.validation_outputs[self.validation_mapping[dataloader_idx]] = outputs

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        # keep new loss weights
        for k, v in self.weights.items():
            if f"weights.{k}" not in state_dict:
                print(f'Add weight {k}: {v.data}')
                state_dict[f'weights.{k}'] = v
                continue
            checkpoint_v = state_dict[f"weights.{k}"]
            if k in self.scheduled_weights or checkpoint_v == v:  # skip scheduled weights or same values
                continue
            print(f'Update weight {k}: {checkpoint_v} --> {v.data}')
            state_dict[f'weights.{k}'] = v
        # remove old loss weights
        remove_keys = []
        for k, v in state_dict.items():
            if 'lambdas' in k or 'scheduled_lambdas' in k:
                print(f'Remove legacy weight key: {k}')
                remove_keys.append(k)
            elif 'weights' in k and k.split('.')[1] not in self.weights.keys():
                print(f'Remove weight: {k}')
                remove_keys.append(k)
        [state_dict.pop(k) for k in remove_keys]

        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs

def _normalize_date(date):
    if date is None:
        return None
    if hasattr(date, 'to_datetime'):
        date = date.to_datetime()
    if hasattr(date, 'isot'):
        return date.isot
    if hasattr(date, 'isoformat'):
        return date.isoformat()
    return str(date)


def _date_from_wcs(wcs):
    if wcs is None:
        return None
    date = getattr(getattr(wcs, 'wcs', None), 'dateobs', None)
    return _normalize_date(date)


def _iter_values(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _save_state_date(data_module):
    data_config = getattr(data_module, 'config', {})
    for wcs in _iter_values(data_config.get('wcs')):
        date = _date_from_wcs(wcs)
        if date is not None:
            return date

    for dataset in getattr(data_module, 'datasets', {}).values():
        for wcs in _iter_values(getattr(dataset, 'wcs', None)):
            date = _date_from_wcs(wcs)
            if date is not None:
                return date
        date = _normalize_date(getattr(dataset, 'date', None))
        if date is not None:
            return date
    return None


@rank_zero_only
def save(save_path, nf2, data_module, config):
    import lightning

    try:
        nf2_version = version('nf2')
    except PackageNotFoundError:
        nf2_version = 'unknown'

    save_state = {'format_version': '0.4',
                  'date': _save_state_date(data_module),
                  'software': {
                      'nf2': nf2_version,
                      'torch': torch.__version__,
                      'lightning': lightning.__version__,
                  },
                  'model': nf2.model,
                  'config': config,
                  'data': data_module.config,
                  'validation_dataset_mapping': data_module.validation_dataset_mapping,
                  'transforms': nf2.transform_modules, }
    torch.save(save_state, save_path)
