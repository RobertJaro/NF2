import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from nf2.train.model import BModel, jacobian, VectorPotentialModel


class NF2Module(LightningModule):

    def __init__(self, sampler, cube_shape, dim=256, positional_encoding=False, use_vector_potential=False,
                 lambda_div=0.1, lambda_ff=0.1, decay_iterations=None, meta_path=None, ):
        """Magnetic field extrapolations trainer

        :param dim: number of neurons per layer (8 layers).
        :param positional_encoding: use positional encoding.
        :param use_vector_potential: derive the magnetic field from a vector potential.
        :param lambda_div: weighting parameter for divergence freeness of the simulation.
        :param lambda_ff: weighting parameter for force freeness of the simulation.
        :param decay_iterations: decay weighting for boundary condition (lambda_B=1000) over n iterations to 1.
        :param meta_path: start from a pre-learned simulation state.
        :param work_directory: directory to store scratch data (prepared batches).
        """
        super().__init__()
        # init model
        if use_vector_potential:
            model = VectorPotentialModel(3, dim, pos_encoding=positional_encoding)
        else:
            model = BModel(3, 3, dim, pos_encoding=positional_encoding)
        self.model = model
        self.sampler = sampler
        self.cube_shape = cube_shape

        # load meta state
        if meta_path:
            state_dict = torch.load(meta_path)['model'].state_dict() \
                if meta_path.endswith('nf2') else torch.load(meta_path)['m']
            model.load_state_dict(state_dict)
            logging.info('Loaded meta state: %s' % meta_path)
        # init
        lambda_B = 1000 if decay_iterations else 1

        self.lambda_B = lambda_B
        self.lambda_B_decay = (1 / 1000) ** (1 / decay_iterations) if decay_iterations is not None else 1
        self.lambda_div, self.lambda_ff = lambda_div, lambda_ff
        self.use_vector_potential = use_vector_potential

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = ExponentialLR(self.optimizer, gamma=(5e-5 / 5e-4) ** (1 / 1e5))  # decay over 1e5 iterations

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        boundary_coords, b_true, b_err = batch
        random_coords = self.sampler.load_sample()

        # concatenate boundary and random points
        n_boundary_coords = boundary_coords.shape[0]
        coords = torch.cat([boundary_coords, random_coords], 0)
        coords.requires_grad = True

        # forward step
        b = self.model(coords)

        # compute boundary loss
        boundary_b = b[:n_boundary_coords]
        b_diff = torch.clip(torch.abs(boundary_b - b_true) - b_err, 0)
        b_diff = torch.mean(b_diff.pow(2).sum(-1))

        # compute div and ff loss
        divergence_loss, force_loss = calculate_loss(b, coords)
        divergence_loss, force_loss = divergence_loss.mean(), force_loss.mean()
        loss = b_diff * self.lambda_B + divergence_loss * self.lambda_div + force_loss * self.lambda_ff

        return {'loss': loss, 'b_diff': b_diff, 'div': divergence_loss, 'ff': force_loss}

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # update training parameters and log
        if self.lambda_B > 1:
            self.lambda_B *= self.lambda_B_decay
        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])
        self.log('Lambda B', self.lambda_B)

        # log results to WANDB
        self.log("train/loss", outputs['loss'])
        self.log("Training Loss",
                 {'b_diff': outputs['b_diff'].mean(), 'divergence': outputs['div'].mean(), 'force-free': outputs['ff'].mean(), 'total': outputs['loss'].mean()})

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb, dataloader_idx):
        if dataloader_idx == 0:
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

            return {'b': b.detach(), 'j': j.detach(), 'div': div.detach()}
        if dataloader_idx == 1:
            boundary_coords, b_true, b_err = batch
            boundary_coords.requires_grad = True

            b = self.model(boundary_coords)

            # compute boundary loss
            b_diff = torch.clip(torch.abs(b - b_true) - b_err, 0)
            b_diff = torch.mean(b_diff.pow(2).sum(-1))
            return {'b_diff': b_diff.detach()}
        else:
            raise NotImplementedError('Validation data loader not supported!')

    def validation_epoch_end(self, outputs_list):
        outputs = outputs_list[0]  # unpack data loader 0
        b = torch.cat([o['b'] for o in outputs])
        j = torch.cat([o['j'] for o in outputs])
        div = torch.cat([o['div'] for o in outputs])

        norm = b.pow(2).sum(-1).pow(0.5) * j.pow(2).sum(-1).pow(0.5)
        angle = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / norm
        sig = torch.asin(torch.clip(angle, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)
        sig = torch.abs(sig)
        weighted_sig = np.average(sig.cpu().numpy(), weights=j.pow(2).sum(-1).pow(0.5).cpu().numpy())
        b_norm = b.pow(2).sum(-1).pow(0.5) + 1e-7
        div_loss = (div / b_norm).mean()
        ff_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / b_norm
        ff_loss = ff_loss.mean()

        outputs = outputs_list[1]  # unpack data loader 1
        b_diff = torch.stack([o['b_diff'] for o in outputs]).mean()

        self.plot_sample(b.reshape([*self.cube_shape, 3]).cpu().numpy())

        self.log("Validation B_diff", b_diff)
        self.log("Validation DIV", div_loss)
        self.log("Validation FF", ff_loss)
        self.log("Validation Sigma", weighted_sig)

        return {'progress_bar': {'b_diff': b_diff, 'div': div_loss, 'ff': ff_loss, 'sigma': weighted_sig},
                'log': {'val/b_diff': b_diff, 'val/div': div_loss, 'val/ff': ff_loss, 'val/sig': weighted_sig}}

    def plot_sample(self, b, n_samples=10):
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        heights = np.linspace(0, 1, n_samples) ** 2 * (b.shape[2] - 1)  # more samples from lower heights
        heights = heights.astype(np.int)
        for i in range(3):
            for j, h in enumerate(heights):
                v_min_max = np.max(np.abs(b[:, :, h, i]))
                axs[i, j].imshow(b[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                                 origin='lower')
                axs[i, j].set_axis_off()
        for j, h in enumerate(heights):
            axs[0, j].set_title('%.01f' % h)
        fig.tight_layout()
        wandb.log({"Slices": fig})
        plt.close('all')


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


def save(save_path, model, data_module):
    torch.save({'model': model,
                'cube_shape': data_module.cube_shape,
                'b_norm': data_module.b_norm,
                'spatial_norm': data_module.spatial_norm,
                'meta_info': data_module.meta_info}, save_path)
