import numpy as np
import wandb
from pytorch_lightning import Callback
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.data.util import cartesian_to_spherical, vector_cartesian_to_spherical



class SlicesCallback(Callback):

    def __init__(self, name, cube_shape):
        self.name = name
        self.cube_shape = cube_shape

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.name not in pl_module.validation_outputs:
            return
        outputs = pl_module.validation_outputs[self.name]
        b = outputs['b']
        j = outputs['j']
        coords = outputs['coords']

        b_cube = b.reshape([*self.cube_shape, 3]).cpu().numpy()
        j_cube = j.reshape([*self.cube_shape, 3]).cpu().numpy()
        c_cube = coords.reshape([*self.cube_shape, 3]).cpu().numpy()

        # transform to spherical coordinates
        c_cube = cartesian_to_spherical(c_cube)
        b_cube = vector_cartesian_to_spherical(b_cube, c_cube)

        self.plot_b(b_cube, c_cube)
        self.plot_current(j_cube, c_cube)

    def plot_b(self, b, coords):
        n_samples = b.shape[2]
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        for i in range(3):
            for j in range(n_samples):
                v_min_max = np.max(np.abs(b[:, :, j]))
                extent = [coords[:, :, j, 2].min(), coords[:, :, j, 2].max(), coords[:, :, j, 1].min(), coords[:, :, j, 1].max()]
                extent = np.rad2deg(extent)
                height = coords[:, :, j, 0].mean()
                im = axs[i, j].imshow(b[:, :, j, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                                 origin='lower', extent=extent)
                axs[i, j].set_xlabel('Longitude [deg]')
                axs[i, j].set_ylabel('Latitude [deg]')
                # add locatable colorbar
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='B [G]')
                axs[i, j].set_title(f'{height:.02f} - $B_{["r", "t", "p"][i]}$')
        fig.tight_layout()
        wandb.log({f"{self.name} - B": fig})
        plt.close('all')

    def plot_current(self, j, coords):
        j = (j ** 2).sum(-1) ** 0.5
        n_samples = j.shape[2]
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
        for i in range(n_samples):
            extent = [coords[:, :, i, 2].min(), coords[:, :, i, 2].max(),
                      coords[:, :, i, 1].min(), coords[:, :, i, 1].max()]
            extent = np.rad2deg(extent)
            height = coords[:, :, i, 0].mean()
            im = axs[i].imshow(j[:, :, i].transpose(), cmap='plasma', origin='lower', norm=LogNorm(), extent=extent)
            axs[i].set_xlabel('Longitude [deg]')
            axs[i].set_ylabel('Latitude [deg]')
            # add locatable colorbar
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='J [G/ds]')
            axs[i].set_title(f'{height:.02f} - $|J|$')
        fig.tight_layout()
        wandb.log({f"{self.name} - Current density": fig})
        plt.close('all')
        # plot integrated current density
        j = np.sum(j, axis=2)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(j.transpose(), cmap='plasma', origin='lower', norm=LogNorm(), extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Current density": fig})
        plt.close('all')


class BoundaryCallback(Callback):

    def __init__(self, cube_shape):
        self.name = 'boundary'
        self.cube_shape = cube_shape

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.name not in pl_module.validation_outputs:
            return
        outputs = pl_module.validation_outputs[self.name]
        b = outputs['b'].cpu().numpy()
        b_true = outputs['b_true'].cpu().numpy()
        coords = outputs['coords'].cpu().numpy()

        # transform to spherical coordinates
        coords = cartesian_to_spherical(coords)

        self.plot_b(b, b_true, coords)

    def plot_b(self, b, b_true, coords):
        fig, axs = plt.subplots(3, 2, figsize=(8, 12))

        b_norm = 0.2
        extent = [coords[:, 2].min(), coords[:, 2].max(), coords[:, 1].min(), coords[:, 1].max()]
        extent = np.rad2deg(extent)

        boundary_shape = self.cube_shape['shapes'][0]
        x = coords[:, 2]
        y = coords[:, 1]

        x = ((x - x.min()) / (x.max() - x.min()) * (boundary_shape[1] - 1)).astype(int)
        y = ((y - y.min()) / (y.max() - y.min()) * (boundary_shape[0] - 1)).astype(int)

        img = np.zeros(boundary_shape)
        img[y, x] = b[:, 0]
        im = axs[0, 0].imshow(img, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B_LOS [G]')

        img = np.zeros(boundary_shape)
        img[y, x] = b_true[:, 0]
        im = axs[0, 1].imshow(img, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B_LOS [G]')

        img = np.zeros(boundary_shape)
        img[y, x] = b[:, 1]
        im = axs[1, 0].imshow(img, vmin=0, vmax=b_norm, cmap='gray', origin='lower', extent=extent)
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B_TRV [G]')

        img = np.zeros(boundary_shape)
        img[y, x] = b_true[:, 1]
        im = axs[1, 1].imshow(img, vmin=0, vmax=b_norm, cmap='gray', origin='lower', extent=extent)
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B_TRV [G]')

        img = np.zeros(boundary_shape)
        img[y, x] = b[:, 2]
        im = axs[2, 0].imshow(img, vmin=-np.pi, vmax=np.pi, cmap='gray', origin='lower', extent=extent)
        divider = make_axes_locatable(axs[2, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Azimuth [rad]')

        img = np.zeros(boundary_shape)
        img[y, x] = b_true[:, 2]
        im = axs[2, 1].imshow(img, vmin=-np.pi, vmax=np.pi, cmap='gray', origin='lower', extent=extent)
        divider = make_axes_locatable(axs[2, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Azimuth [rad]')

        [ax.set_xlabel('Longitude [deg]') for ax in np.ravel(axs)]
        [ax.set_ylabel('Latitude [deg]') for ax in np.ravel(axs)]
        fig.tight_layout()
        wandb.log({f"{self.name} - B": fig})
        plt.close('all')