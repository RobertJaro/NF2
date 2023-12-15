import logging

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from nf2.train.loss import *
from nf2.train.model import BModel, jacobian, VectorPotentialModel, HeightMappingModel


class NF2Module(LightningModule):

    def __init__(self, validation_settings, dim=256, boundary_loss="ptr",
                 lambda_b={'start': 1e3, 'end': 1, 'iterations': 1e5}, lambda_divergence=None, lambda_force_free=None,
                 lambda_height=None, lambda_nans=None, lambda_radial=None,
                 lambda_energy_gradient=None, lambda_potential=None,
                 lr_params={"start": 5e-4, "end": 5e-5, "iterations": 1e5},
                 meta_path=None,
                 use_positional_encoding=False, use_vector_potential=False, **kwargs):
        """
        The main module for training the neural field model.

        :param validation_settings: A dictionary containing the validation settings.
        :param dim: The dimension of the model.
        :param lambda_b: The lambda for the magnetic field loss.
        :param lambda_divergence: The lambda for the divergence loss.
        :param lambda_force_free: The lambda for the force-free loss.
        :param lambda_height: The lambda for the height loss.
        :param lambda_nans: The lambda for the NaN loss.
        :param lambda_radial: The lambda for the radial loss.
        :param lambda_energy_gradient: The lambda for the energy gradient loss.
        :param lambda_potential: The lambda for the potential loss.
        :param lr_params: The learning rate parameters.
        :param meta_path: The path to the meta state.
        :param use_positional_encoding: Whether to use positional encoding. Default: None.
        :param use_vector_potential: Whether to use the vector potential model.
        """
        super().__init__()
        # init model
        if use_vector_potential:
            model = VectorPotentialModel(3, dim, pos_encoding=use_positional_encoding)
        else:
            model = BModel(3, 3, dim, pos_encoding=use_positional_encoding)
        # load meta state
        if meta_path:
            state_dict = torch.load(meta_path)['model'].state_dict() \
                if meta_path.endswith('nf2') else torch.load(meta_path)['m']
            model.load_state_dict(state_dict)
            logging.info('Loaded meta state: %s' % meta_path)

        # init boundary loss module
        if boundary_loss == "azimuth":
            boundary_loss_module = AzimuthBoundaryLoss
            self.b_transform = AzimuthTransform()
        elif boundary_loss == "ptr":
            boundary_loss_module = BoundaryLoss
            self.b_transform = SphericalTransform()
        else:
            raise ValueError(f"Invalid boundary loss: {boundary_loss}, must be in ['azimuth', 'rtp']")
        # mapping
        lambda_mapping = {'b': lambda_b, 'divergence': lambda_divergence, 'force-free': lambda_force_free,
                          'potential': lambda_potential, 'height': lambda_height, 'NaNs': lambda_nans,
                          'radial': lambda_radial, 'energy_gradient': lambda_energy_gradient, }
        loss_module_mapping = {'b': boundary_loss_module,
                               'divergence': DivergenceLoss, 'force-free': ForceFreeLoss, 'potential': PotentialLoss,
                               'height': HeightLoss, 'NaNs': NaNLoss, 'radial': RadialLoss,
                               'energy_gradient': EnergyGradientLoss}
        # init lambdas and loss modules
        scheduled_lambdas = {}
        lambdas = {}
        loss_modules = {}
        for k, v in lambda_mapping.items():
            if v is None:
                continue
            if isinstance(v, dict):
                value = torch.tensor(v['start'], dtype=torch.float32)
                gamma = torch.tensor((v['end'] / v['start']) ** (1 / v['iterations']), dtype=torch.float32)
                end = torch.tensor(v['end'], dtype=torch.float32)
                scheduled_lambdas[k] = {'gamma': gamma, 'end': end}
            else:
                value = torch.tensor(v, dtype=torch.float32)
            lambdas[k] = value
            loss_modules[k] = loss_module_mapping[k]()

        self.model = model
        self.validation_settings = validation_settings
        assert 'boundary' not in validation_settings['names'], "'boundary' is a reserved callback name!"
        self.lr_params = lr_params
        self.scheduled_lambdas = nn.ParameterDict(scheduled_lambdas)
        self.lambdas = nn.ParameterDict(lambdas)
        #
        self.use_vector_potential = use_vector_potential
        self.validation_outputs = {}
        self.loss_modules = nn.ModuleDict(loss_modules)

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        if isinstance(self.lr_params, dict):
            lr_start = self.lr_params['start']
            lr_end = self.lr_params['end']
            iterations = self.lr_params['iterations']
        else:
            lr_start = self.lr_params
            lr_end = self.lr_params
            iterations = 1
            self.lr_params = {'start': lr_start, 'end': lr_end, 'iterations': iterations}
        self.optimizer = torch.optim.Adam(parameters, lr=lr_start)
        self.scheduler = ExponentialLR(self.optimizer, gamma=(lr_end / lr_start) ** (1 / iterations))

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        boundary_batch = batch['boundary']
        boundary_coords = boundary_batch['coords']
        b_true = boundary_batch['values']
        transform = boundary_batch['transform'] if 'transform' in boundary_batch else None
        b_err = boundary_batch['errors'] if 'errors' in boundary_batch else 0

        boundary_coords.requires_grad = True

        random_coords = batch['random']
        random_coords.requires_grad = True

        # concatenate boundary and random points
        n_boundary_coords = boundary_coords.shape[0]
        coords = torch.cat([boundary_coords, random_coords], 0)

        # forward step
        b = self.model(coords)

        # compute derivatives
        jac_matrix = jacobian(b, coords)

        state_dict = {
            "b": b, "b_true": b_true, "b_err": b_err, "transform": transform, "coords": coords,
            "n_boundary_coords": n_boundary_coords,
            "boundary_coords": coords[:n_boundary_coords], "random_coords": coords[n_boundary_coords:],
            "boundary_b": b[:n_boundary_coords],  "random_b": b[n_boundary_coords:],
            "jac_matrix": jac_matrix,
        }
        loss_dict = {k: module(**state_dict) for k, module in self.loss_modules.items()}
        total_loss = sum([self.lambdas[k] * loss_dict[k] for k in loss_dict.keys()])

        loss_dict['loss'] = total_loss
        return loss_dict

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update lambda parameters and log
        for k in self.scheduled_lambdas.keys():
            param = self.lambdas[k]
            gamma = self.scheduled_lambdas[k]['gamma']
            if (gamma < 1 and param > self.scheduled_lambdas[k]['end']) or \
                    (gamma > 1 and param < self.scheduled_lambdas[k]['end']):
                new_lambda = param * gamma
                param.copy_(new_lambda)
            self.log('lambda_' + k, self.lambdas[k])
        # update learning rate
        if self.scheduler.get_last_lr()[0] > self.lr_params['end']:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

        # log results to WANDB
        self.log("train", {k: v.mean() for k, v in outputs.items()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx):
        if dataloader_idx == 0:
            boundary_coords = batch['coords']
            b_true = batch['values']
            b_err = batch['errors'] if 'errors' in batch else 0
            boundary_coords.requires_grad = True
            transform = batch['transform'] if 'transform' in batch else None

            b = self.model(boundary_coords)
            b, b_true = self.b_transform(b, b_true, transform=transform)

            # compute diff
            b_diff_error = torch.clip(torch.abs(b - b_true) - b_err, 0)
            b_diff_error = torch.mean(torch.nansum(b_diff_error.pow(2), -1).pow(0.5))

            b_diff = torch.abs(b - b_true)
            b_diff = torch.mean(torch.nansum(b_diff.pow(2), -1).pow(0.5))

            return {'b_diff_error': b_diff_error.detach(), 'b_diff': b_diff.detach(),
                    'b': b.detach(), 'b_true': b_true.detach(), 'coords': boundary_coords.detach()}
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
        self.validation_outputs = {}  # reset validation outputs
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps
        outputs = outputs_list[1]  # unpack data loader 1
        # stack magnetic field and unnormalize
        b = torch.cat([o['b'] for o in outputs]) * self.validation_settings['gauss_per_dB']
        j = torch.cat([o['j'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
            self.validation_settings['Mm_per_ds']
        div = torch.cat([o['div'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
              self.validation_settings['Mm_per_ds']

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
        b = torch.cat([o['b'] for o in outputs])
        b_true = torch.cat([o['b_true'] for o in outputs])
        coords = torch.cat([o['coords'] for o in outputs])
        self.validation_outputs['boundary'] = {'b': b, 'b_true': b_true, 'coords': coords}

        for name, outputs in zip(self.validation_settings['names'], outputs_list[2:]):
            b = torch.cat([o['b'] for o in outputs]) * self.validation_settings['gauss_per_dB']
            j = torch.cat([o['j'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
                self.validation_settings['Mm_per_ds']
            div = torch.cat([o['div'] for o in outputs]) * self.validation_settings['gauss_per_dB'] / \
                  self.validation_settings['Mm_per_ds']
            coords = torch.cat([o['coords'] for o in outputs])
            self.validation_outputs[name] = {'b': b, 'j': j, 'div': div, 'coords': coords}

        self.log("valid", {"b": b_diff, "b_error": b_diff_error,
                           "divergence": div_loss, "force-free": ff_loss, "$\sigma_J$": weighted_sig})

        return {'progress_bar': {'b': b_diff, 'divergence': div_loss, 'force-free': ff_loss, 'sigma': weighted_sig}}

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

        self.load_state_dict(state_dict, strict=False)

def save(save_path, model, data_module, config):
    save_state = {'model': model,
                  'cube_shape': data_module.cube_shape,
                  'b_norm': data_module.b_norm,
                  'spatial_norm': data_module.spatial_norm,
                  'meta_data': data_module.meta_data,
                  'config': config,
                  'height_mapping': data_module.height_mapping if hasattr(data_module, 'height_mapping') else None,
                  'Mm_per_pixel': data_module.Mm_per_pixel if hasattr(data_module, 'Mm_per_pixel') else None,
                  }
    torch.save(save_state, save_path)
