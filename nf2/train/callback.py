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
                # add locatable colorbar
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
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
            # add locatable colorbar
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
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