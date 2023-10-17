import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.colors import LogNorm
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nf2.data.util import vector_cartesian_to_spherical, cartesian_to_spherical
from nf2.train.model import BModel, jacobian, VectorPotentialModel, HeightMappingModel


class NF2Module(LightningModule):

    def __init__(self, validation_settings, dim=256, lambda_b={'start': 1e3, 'end': 1, 'iterations': 1e5},
                 lambda_div=0.1, lambda_ff=0.1,
                 lambda_height_reg=0, lambda_min_energy_nans=0, lambda_min_energy=0, lambda_radial_reg=0,
                 lambda_energy_gradient_reg = 0,
                 lr_params={"start": 5e-4, "end": 5e-5, "decay_iterations": 1e5}, meta_path=None,
                 use_positional_encoding=False, use_vector_potential=False, use_height_mapping=False, spherical=False,
                 **kwargs):
        """Magnetic field extrapolations trainer

        :param validation_settings: validation settings
        :param dim: model dimension
        :param lambda_b: lambda for the magnetic field loss
        :param lambda_div: lambda for the divergence loss
        :param lambda_ff: lambda for the force-free loss
        :param lambda_height_reg: lambda for the height regularization loss
        :param lambda_min_energy_nans: lambda for the minimum energy loss (NaNs)
        :param lambda_min_energy: lambda for the minimum energy loss
        :param lambda_radial_reg: lambda for the radial regularization loss
        :param lr_params: learning rate parameters
        :param meta_path: path to a meta state
        :param use_positional_encoding: use positional encoding
        :param use_vector_potential: use vector potential model
        :param use_height_mapping: use height mapping model
        :param spherical: use spherical coordinates
        :param kwargs: additional arguments
        """
        super().__init__()
        # init model
        if use_vector_potential:
            model = VectorPotentialModel(3, dim, pos_encoding=use_positional_encoding)
        else:
            model = BModel(3, 3, dim, pos_encoding=use_positional_encoding)
        if use_height_mapping:
            self.height_mapping_model = HeightMappingModel(3, dim)
        else:
            self.height_mapping_model = None
        self.model = model
        self.validation_settings = validation_settings
        self.lr_params = lr_params

        # load meta state
        if meta_path:
            state_dict = torch.load(meta_path)['model'].state_dict() \
                if meta_path.endswith('nf2') else torch.load(meta_path)['m']
            model.load_state_dict(state_dict)
            logging.info('Loaded meta state: %s' % meta_path)
        # init lambdas
        scheduled_lambdas = {}
        lambdas = {}
        for k, v in {'lambda_b': lambda_b, 'lambda_div': lambda_div, 'lambda_ff': lambda_ff,
                     'lambda_height_reg': lambda_height_reg, 'lambda_min_energy_nans': lambda_min_energy_nans,
                     'lambda_min_energy': lambda_min_energy, 'lambda_radial_reg': lambda_radial_reg,
                     'lambda_energy_gradient_reg': lambda_energy_gradient_reg}.items():
            if isinstance(v, dict):
                value = torch.tensor(v['start'], dtype=torch.float32)
                gamma = torch.tensor((v['end'] / v['start']) ** (1 / v['iterations']), dtype=torch.float32)
                end = torch.tensor(v['end'], dtype=torch.float32)
                scheduled_lambdas[k] = {'gamma': gamma, 'end': end}
            else:
                value = torch.tensor(v, dtype=torch.float32)
            lambdas[k] = value

        self.scheduled_lambdas = nn.ParameterDict(scheduled_lambdas)
        self.lambdas = nn.ParameterDict(lambdas)
        #
        self.use_vector_potential = use_vector_potential
        self.use_height_mapping = use_height_mapping
        self.spherical = spherical
        self.validation_outputs = {}

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        if self.use_height_mapping:
            parameters += list(self.height_mapping_model.parameters())
        if isinstance(self.lr_params, dict):
            lr_start = self.lr_params['start']
            lr_end = self.lr_params['end']
            decay_iterations = self.lr_params['decay_iterations']
        else:
            lr_start = self.lr_params
            lr_end = self.lr_params
            decay_iterations = 1
        self.optimizer = torch.optim.Adam(parameters, lr=lr_start)
        self.scheduler = ExponentialLR(self.optimizer, gamma=(lr_end / lr_start) ** (1 / decay_iterations))

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        boundary_batch = batch['boundary']
        boundary_coords = boundary_batch['coords']
        b_true = boundary_batch['values']
        transform = boundary_batch['transform'] if 'transform' in boundary_batch else None
        b_err = boundary_batch['errors'] if 'errors' in boundary_batch else 0
        boundary_coords.requires_grad = True
        if self.use_height_mapping:
            original_coords = boundary_coords
            boundary_coords = self.height_mapping_model(boundary_coords, boundary_batch['height_ranges'])

        random_coords = batch['random']
        random_coords.requires_grad = True

        # concatenate boundary and random points
        n_boundary_coords = boundary_coords.shape[0]
        coords = torch.cat([boundary_coords, random_coords], 0)

        # forward step
        b = self.model(coords)

        # compute boundary loss
        boundary_b = b[:n_boundary_coords]
        transformed_b = torch.einsum('ijk,ik->ij', transform, boundary_b) if transform is not None else boundary_b
        b_diff = torch.clip(torch.abs(transformed_b - b_true) - b_err, 0)
        b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1))

        # minimum energy regularization for nan components
        if torch.isnan(b_true).sum() == 0:
            min_energy_NaNs_regularization = torch.zeros((1,), device=b_true.device)
        else:
            min_energy_NaNs_regularization = boundary_b[torch.isnan(b_true)].pow(2).sum() / torch.isnan(b_true).sum()

        # radial distance weight
        radius_weight = random_coords.pow(2).sum(-1).pow(0.5) - 1

        # min energy regularization
        min_energy_regularization = torch.mean(torch.nansum(b[n_boundary_coords:].pow(2), -1) * radius_weight)

        # radial regularization
        normalization = torch.norm(b[n_boundary_coords:], dim=-1) * torch.norm(random_coords, dim=-1) + 1e-8
        dot_product = (b[n_boundary_coords:] * random_coords).sum(-1)
        radial_regularization = 1 - (dot_product / normalization).abs()
        radial_regularization = (radial_regularization * radius_weight).mean()

        # compute div and ff loss
        divergence_loss, force_loss  = calculate_loss(b, coords)
        divergence_loss, force_loss = divergence_loss.mean(), force_loss.mean()

        # magnetic energy radial decrease
        E = b[n_boundary_coords:].pow(2).sum(-1)
        dE_dxyz = torch.autograd.grad(E[:, None], random_coords,
                                      grad_outputs=torch.ones_like(E[:, None]).to(E),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
        coords_spherical = cartesian_to_spherical(random_coords, f=torch)
        t = coords_spherical[:, 1]
        p = coords_spherical[:, 2]
        dE_dr = (torch.sin(t) * torch.cos(p)) * dE_dxyz[:, 0] + \
                (torch.sin(t) * torch.sin(p)) * dE_dxyz[:, 1] + \
                torch.cos(p) * dE_dxyz[:, 2]
        energy_gradient_regularization = (torch.relu(dE_dr) * radius_weight).mean()

        loss = b_diff * self.lambdas['lambda_b'] + \
               divergence_loss * self.lambdas['lambda_div'] + \
               force_loss * self.lambdas['lambda_ff'] + \
               min_energy_NaNs_regularization * self.lambdas['lambda_min_energy_nans'] + \
               min_energy_regularization * self.lambdas['lambda_min_energy'] + \
               radial_regularization * self.lambdas['lambda_radial_reg'] + \
               energy_gradient_regularization * self.lambdas['lambda_energy_gradient_reg']


        loss_dict = {'b_diff': b_diff, 'divergence': divergence_loss, 'force-free': force_loss,
                     'min_energy_NaNs_regularization': min_energy_NaNs_regularization,
                     'min_energy_regularization': min_energy_regularization,
                     'radial_regularization': radial_regularization,
                     'energy_gradient_regularization': energy_gradient_regularization}
        if self.use_height_mapping:
            height_diff = torch.abs(boundary_coords[:, 2] - original_coords[:, 2])
            normalization = (boundary_batch['height_ranges'][:, 1] - boundary_batch['height_ranges'][:, 0]) + 1e-6
            height_regularization = torch.true_divide(height_diff, normalization).mean()
            loss += self.lambdas['lambda_height_reg'] * height_regularization
            loss_dict['height_regularization'] = height_regularization

        loss_dict['loss'] = loss
        return loss_dict

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update lambda parameters and log
        for k in self.scheduled_lambdas.keys():
            param = self.lambdas[k]
            if param > self.scheduled_lambdas[k]['end']:
                new_lambda = param * self.scheduled_lambdas[k]['gamma']
                param.copy_(new_lambda)
            self.log(k, self.lambdas[k])
        # update learning rate
        if self.scheduler.get_last_lr()[0] > self.lr_params['end']:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

        # log results to WANDB
        self.log("train/loss", outputs['loss'])
        self.log("Training Loss", {k: v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx):
        if dataloader_idx == 0:
            boundary_coords = batch['coords']
            b_true = batch['values']
            b_err = batch['errors'] if 'errors' in batch else 0
            boundary_coords.requires_grad = True
            transform = batch['transform'] if 'transform' in batch else None
            if self.use_height_mapping:
                boundary_coords = self.height_mapping_model(boundary_coords, batch['height_ranges'])
            b = self.model(boundary_coords)
            b = torch.einsum('ijk,ik->ij', transform, b) if transform is not None else b

            # compute boundary loss
            b_diff_error = torch.clip(torch.abs(b - b_true) - b_err, 0)
            b_diff_error = torch.mean(torch.nansum(b_diff_error.pow(2), -1).pow(0.5))
            b_diff = torch.abs(b - b_true)
            b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1).pow(0.5))
            return {'b_diff_error': b_diff_error.detach(), 'b_diff': b_diff.detach()}
        else:
            coords = batch
            coords.requires_grad = True

            # forward step
            b = self.model(coords)

            jac_matrix = jacobian(b, coords)
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

            return {'b': b.detach(), 'j': j.detach(), 'div': div.detach(), 'coords': coords.detach()}

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps
        outputs = outputs_list[1]  # unpack data loader 1
        # stack magnetic field and unnormalize
        b = torch.cat([o['b'] for o in outputs]) * self.validation_settings['gauss_per_dB']
        j = torch.cat([o['j'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
            self.validation_settings['Mm_per_ds']
        div = torch.cat([o['div'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
              self.validation_settings['Mm_per_ds']
        coords = torch.cat([o['coords'] for o in outputs])

        norm = b.pow(2).sum(-1).pow(0.5) * j.pow(2).sum(-1).pow(0.5)
        angle = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / norm
        sig = torch.asin(torch.clip(angle, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)
        sig = torch.abs(sig)
        weighted_sig = np.average(sig.cpu().numpy(), weights=j.pow(2).sum(-1).pow(0.5).cpu().numpy())
        b_norm = b.pow(2).sum(-1).pow(0.5) + 1e-7
        div_loss = (div / b_norm).mean()
        ff_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / b_norm
        ff_loss = ff_loss.mean()

        outputs = outputs_list[0]  # unpack data loader 0
        b_diff = torch.stack([o['b_diff'] for o in outputs]).mean() * self.validation_settings['gauss_per_dB']
        b_diff_error = torch.stack([o['b_diff_error'] for o in outputs]).mean() * self.validation_settings[
            'gauss_per_dB']

        self.validation_outputs = {} # reset validation outputs
        for name, outputs in zip(self.validation_settings['names'], outputs_list[2:]):
            b = torch.cat([o['b'] for o in outputs]) * self.validation_settings['gauss_per_dB']
            j = torch.cat([o['j'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
                self.validation_settings['Mm_per_ds']
            div = torch.cat([o['div'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
                  self.validation_settings['Mm_per_ds']
            coords = torch.cat([o['coords'] for o in outputs])
            self.validation_outputs[name] = {'b': b, 'j': j, 'div': div, 'coords': coords}

        self.log("Validation B_diff", b_diff)
        self.log("Validation B_diff_error", b_diff_error)
        self.log("Validation DIV", div_loss)
        self.log("Validation FF", ff_loss)
        self.log("Validation Sigma", weighted_sig)

        return {'progress_bar': {'b_diff': b_diff, 'div': div_loss, 'ff': ff_loss, 'sigma': weighted_sig},
                'log': {'val/b_diff': b_diff, 'val/div': div_loss, 'val/ff': ff_loss, 'val/sig': weighted_sig}}

    def on_load_checkpoint(self, checkpoint):
        # keep new lambdas
        for k, v in self.lambdas.items():
            checkpoint_v = checkpoint["state_dict"][f"lambdas.{k}"]
            if k in self.scheduled_lambdas or checkpoint_v == v: # skip scheduled lambdas or same values
                continue
            print(f'Update lambda {k}: {checkpoint_v} --> {v.data}')
            checkpoint['state_dict'][f'lambdas.{k}'] = v
        super().on_load_checkpoint(checkpoint)

def calculate_loss(b, coords):
    jac_matrix = jacobian(b, coords)
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
    jxb = torch.cross(j, b, -1)
    force_loss = torch.sum(jxb ** 2, dim=-1) / (torch.sum(b ** 2, dim=-1) + 1e-7)
    divergence_loss = (dBx_dx + dBy_dy + dBz_dz) ** 2
    return divergence_loss, force_loss


def save(save_path, model, data_module, config, height_mapping_model=None):
    save_state = {'model': model,
                  'cube_shape': data_module.cube_shape,
                  'b_norm': data_module.b_norm,
                  'spatial_norm': data_module.spatial_norm,
                  'meta_data': data_module.meta_data,
                  'config': config,
                  'height_mapping_model': height_mapping_model,
                  'height_mapping': data_module.height_mapping if hasattr(data_module, 'height_mapping') else None,
                  'Mm_per_pixel': data_module.Mm_per_pixel if hasattr(data_module, 'Mm_per_pixel') else None,
                  }
    torch.save(save_state, save_path)
