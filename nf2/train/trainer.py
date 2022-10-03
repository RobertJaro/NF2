import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.cuda import get_device_name
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from nf2.data.dataset import ImageDataset
from nf2.data.loader import load_hmi_dataset, RandomSampler
from nf2.train.model import CubeModel, jacobian, VectorPotentialModel


class NF2Trainer:

    def __init__(self, base_path, b_cube, error_cube, height, spatial_norm, b_norm, batch_size, dim=256,
                 positional_encoding=False, meta_path=None, potential_boundary=True, use_vector_potential=False,
                 lambda_div=0.1, lambda_ff=0.1, decay_epochs=None,
                 device=None, work_directory=None):
        self.base_path = base_path
        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm
        self.batch_size = batch_size

        work_directory = base_path if work_directory is None else work_directory

        os.makedirs(base_path, exist_ok=True)
        os.makedirs(work_directory, exist_ok=True)

        self.save_path = os.path.join(base_path, 'extrapolation_result.nf2')
        self.checkpoint_path = os.path.join(base_path, 'checkpoint.pt')

        # set logging
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        log.addHandler(logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")))  # set the new file handler
        log.addHandler(logging.StreamHandler())  # set the new console handler

        # log settings
        logging.info('Configuration:')
        logging.info('dim: %d, lambda_div: %f, lambda_ff: %f, potential: %s, decay_epochs: %s' % (
            dim, lambda_div, lambda_ff, str(potential_boundary), str(decay_epochs)))

        self.b_cube = b_cube
        self.error_cube = error_cube

        n_gpus = torch.cuda.device_count()
        device_names = [get_device_name(i) for i in range(n_gpus)]
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('Using device: %s (gpus %d) %s' % (str(device), n_gpus, str(device_names)))

        # load dataset
        self.dataset = load_hmi_dataset(b_cube, error_cube, height, spatial_norm, b_norm, work_directory, batch_size,
                                        plot=True, plot_path=base_path, potential_boundary=potential_boundary)
        self.cube_shape = [*b_cube.shape[:-1], height]
        self.sampler = RandomSampler(self.cube_shape, spatial_norm, batch_size * 2)

        # init model
        if use_vector_potential:
            model = VectorPotentialModel(3, dim, pos_encoding=positional_encoding)
        else:
            model = CubeModel(3, 3, dim, pos_encoding=positional_encoding)
        parallel_model = nn.DataParallel(model)
        parallel_model.to(device)
        opt = torch.optim.Adam(parallel_model.parameters(), lr=5e-4)

        # load last state
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=device)
            init_epoch = state_dict['epoch']
            model.load_state_dict(state_dict['m'])
            opt.load_state_dict(state_dict['o'])
            history = state_dict['history']
            lambda_B = state_dict['lambda_B']
            logging.info('Resuming training from epoch %d' % init_epoch)
        else:
            init_epoch = 0
            lambda_B = 1000 if decay_epochs else 1
            if meta_path:
                state_dict = torch.load(meta_path, map_location=device)['model'].state_dict() if meta_path.endswith(
                    'nf2') \
                    else torch.load(meta_path, map_location=device)['m']
                model.load_state_dict(state_dict)
                lambda_B = 1
                opt = torch.optim.Adam(parallel_model.parameters(), lr=5e-5)
                logging.info('Loaded meta state: %s' % meta_path)
            history = {'epoch': [], 'height': [],
                       'b_loss': [], 'divergence_loss': [], 'force_loss': [], 'sigma_angle': []}

        scheduler = ExponentialLR(opt, gamma=(5e-5 / 5e-4) ** (1 / 700))
        self.model = model
        self.parallel_model = parallel_model
        self.device = device
        self.opt = opt
        self.scheduler = scheduler
        self.init_epoch = init_epoch
        self.history = history
        self.lambda_B = lambda_B
        self.decay_epochs = decay_epochs
        self.lambda_div, self.lambda_ff = lambda_div, lambda_ff

    def train(self, epochs, log_interval=100, validation_interval=100, num_workers=None):
        start_time = datetime.now()

        num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=num_workers, pin_memory=True, shuffle=True)

        model = self.parallel_model
        opt = self.opt
        history = self.history
        device = self.device
        lambda_div, lambda_ff = self.lambda_div, self.lambda_ff
        lambda_B_decay = (1 / 1000) ** (1 / self.decay_epochs) if self.decay_epochs is not None else 1

        model.train()
        for epoch in range(self.init_epoch, epochs):
            total_b_diff = []
            total_divergence_loss = []
            total_force_loss = []

            for boundary_coords, b_true, b_err in tqdm(data_loader):
                opt.zero_grad()
                boundary_coords, b_true, b_err = boundary_coords.to(device), b_true.to(device), b_err.to(device)
                random_coords = self.sampler.load_sample()

                n_boundary_coords = boundary_coords.shape[0]
                coords = torch.cat([boundary_coords, random_coords], 0)
                coords.requires_grad = True
                b = model(coords)

                boundary_b = b[:n_boundary_coords]
                b_diff = torch.clip(torch.abs(boundary_b - b_true) - b_err, 0)
                b_diff = torch.mean(b_diff.pow(2).sum(-1))

                divergence_loss, force_loss = calculate_loss(b, coords)

                opt.zero_grad()  # reset grad from auto-gradient operation
                (b_diff * self.lambda_B +
                 divergence_loss.mean() * lambda_div +
                 force_loss.mean() * lambda_ff).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step()

                total_b_diff += [b_diff.detach().cpu().numpy()]
                total_divergence_loss += [divergence_loss.mean().detach().cpu().numpy()]
                total_force_loss += [force_loss.mean().detach().cpu().numpy()]
            logging.info('[Epoch %05d/%05d] [B-Field: %.08f; Div: %.08f; For: %.08f] [%s]' %
                         (epoch + 1, epochs,
                          np.mean(total_b_diff),
                          np.mean(total_divergence_loss),
                          np.mean(total_force_loss),
                          datetime.now() - start_time))
            if self.lambda_B > 1:
                self.lambda_B *= lambda_B_decay
            if self.scheduler.get_last_lr()[0] > 5e-5:
                self.scheduler.step()
            if log_interval > 0 and (epoch + 1) % log_interval == 0:
                model.eval()
                self.plot_sample(epoch)
                model.train()
                logging.info('Lambda B: %f' % (self.lambda_B))
                logging.info('LR: %f' % (self.scheduler.get_last_lr()[0]))
            if validation_interval > 0 and (epoch + 1) % validation_interval == 0:
                model.eval()
                self.save(epoch)
                # validate and plot
                mean_b, total_divergence, mean_force, sigma_angle = self.validate(self.height)
                logging.info('Validation [Cube: B: %.03f; Div: %.03f; For: %.03f; Sig: %.03f]' %
                             (mean_b, total_divergence, mean_force, sigma_angle))
                #
                history['epoch'].append(epoch + 1)
                history['b_loss'].append(mean_b.mean())
                history['divergence_loss'].append(total_divergence)
                history['force_loss'].append(mean_force)
                history['sigma_angle'].append(sigma_angle)
                self.plotHistory()
                #
                model.train()
        # save final model state
        torch.save({'m': self.model.state_dict(),
                    'o': self.opt.state_dict(), },
                   os.path.join(self.base_path, 'final.pt'))
        torch.save({'model': self.model,
                    'cube_shape': self.cube_shape,
                    'normalization': self.b_norm,
                    'spatial_normalization': self.spatial_norm}, self.save_path)
        return self.save_path

    def save(self, epoch):
        torch.save({'model': self.model,
                    'cube_shape': self.cube_shape,
                    'normalization': self.b_norm,
                    'spatial_normalization': self.spatial_norm}, self.save_path)
        torch.save({'epoch': epoch + 1,
                    'm': self.model.state_dict(),
                    'o': self.opt.state_dict(),
                    'history': self.history,
                    'lambda_B': self.lambda_B},
                   self.checkpoint_path)

    def plot_sample(self, epoch, n_samples=10):
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        heights = np.linspace(0, 1, n_samples) ** 2 * (self.height - 1)  # more samples from lower heights
        imgs = np.array([self.get_image(h) for h in heights])
        for i in range(3):
            for j in range(10):
                v_min_max = np.max(np.abs(imgs[j, ..., i]))
                axs[i, j].imshow(imgs[j, ..., i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                                 origin='lower')
                axs[i, j].set_axis_off()
        for j, h in enumerate(heights):
            axs[0, j].set_title('%.01f' % h)
        fig.tight_layout()
        fig.savefig(os.path.join(self.base_path, '%05d.jpg' % (epoch + 1)))
        plt.close(fig)

    def get_image(self, z=0):
        image_loader = DataLoader(ImageDataset([*self.cube_shape, 3], self.spatial_norm, z),
                                  batch_size=self.batch_size, shuffle=False)
        image = []
        for coord in image_loader:
            coord.requires_grad = True
            pred_pix = self.model(coord.to(self.device))
            image.extend(pred_pix.detach().cpu().numpy())
        image = np.array(image).reshape((*self.cube_shape[:2], 3))
        return image

    def validate(self, z):
        b, j, div, coords = self.get_cube(z, self.batch_size)
        b = b.unsqueeze(0) * self.b_norm
        j = j.unsqueeze(0) * self.b_norm / self.spatial_norm
        div = div.unsqueeze(0) * self.b_norm / self.spatial_norm

        norm = b.pow(2).sum(-1).pow(0.5) * j.pow(2).sum(-1).pow(0.5)
        angle = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / norm
        sig = torch.asin(torch.clip(angle, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)
        sig = torch.abs(sig)
        weighted_sig = np.average(sig.numpy(), weights=j.pow(2).sum(-1).pow(0.5).numpy())

        b_diff = torch.abs(b[0, :, :, 0, :] - self.b_cube) - self.error_cube
        b_diff = torch.clip(b_diff, 0, None)
        b_diff = torch.sqrt((b_diff ** 2).sum(-1))

        b_norm = b.pow(2).sum(-1).pow(0.5) + 1e-7
        div_loss = div / b_norm
        for_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / b_norm

        return b_diff.mean().numpy(), torch.mean(div_loss).numpy(), \
               torch.mean(for_loss).numpy(), weighted_sig

    def get_cube(self, z, batch_size=int(1e4)):
        b = []
        j = []
        div = []

        coords = np.stack(np.mgrid[:self.cube_shape[0], :self.cube_shape[1], :z], -1)
        coords = torch.tensor(coords / self.spatial_norm, dtype=torch.float32)
        coords_shape = coords.shape[:-1]
        coords = coords.view((-1, 3))
        for k in tqdm(range(int(np.ceil(coords.shape[0] / batch_size)))):
            coord = coords[k * batch_size: (k + 1) * batch_size]
            coord.requires_grad = True
            coord = coord.to(self.device)
            b_batch = self.model(coord)

            jac_matrix = jacobian(b_batch, coord)
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
            j_batch = torch.stack([rot_x, rot_y, rot_z], -1)
            div_batch = torch.abs(dBx_dx + dBy_dy + dBz_dz)
            #
            b += [b_batch.detach().cpu()]
            j += [j_batch.detach().cpu()]
            div += [div_batch.detach().cpu()]

        b = torch.cat(b, dim=0).view((*coords_shape, 3))
        j = torch.cat(j, dim=0).view((*coords_shape, 3))
        div = torch.cat(div, dim=0).view(coords_shape)
        return b, j, div, coords

    def plotHistory(self):
        history = self.history
        plt.figure(figsize=(12, 16))
        plt.subplot(411)
        plt.plot(history['epoch'], history['b_loss'], label='B')
        plt.xlabel('Epoch')
        plt.ylabel('B')
        plt.subplot(412)
        plt.plot(history['epoch'], history['divergence_loss'], label='Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Divergence')
        plt.subplot(413)
        plt.plot(history['epoch'], history['force_loss'], label='Force')
        plt.xlabel('Epoch')
        plt.ylabel('Force')
        plt.subplot(414)
        plt.plot(history['epoch'], history['sigma_angle'], label='Angle')
        plt.xlabel('Epoch')
        plt.ylabel('Angle')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_path, 'history.jpg'))
        plt.close()


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
