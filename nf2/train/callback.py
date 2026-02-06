import numpy as np
import torch
import wandb
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from nf2.data.util import cartesian_to_spherical, vector_cartesian_to_spherical, img_to_los_trv_azi, los_trv_azi_to_img


class SphericalSlicesCallback(Callback):

    def __init__(self, name, cube_shape, gauss_per_dB, Mm_per_ds):
        self.name = name
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.name not in pl_module.validation_outputs:
            return
        outputs = pl_module.validation_outputs[self.name]
        b = outputs['b'] * self.gauss_per_dB
        j = outputs['j'] * self.gauss_per_dB / self.Mm_per_ds
        coords = outputs['coords']

        b_cube = b.reshape([*self.cube_shape, 3]).cpu().numpy()
        j_cube = j.reshape([*self.cube_shape, 3]).cpu().numpy()
        c_cube = coords.reshape([*self.cube_shape, 3]).cpu().numpy()

        # transform to spherical coordinates
        c_cube = cartesian_to_spherical(c_cube)
        b_cube = vector_cartesian_to_spherical(b_cube, c_cube)

        c_cube[..., 0] *= self.Mm_per_ds / (1 * u.solRad).to_value(u.Mm)

        self.plot_b(b_cube, c_cube)
        self.plot_current(j_cube, c_cube)

    def plot_b(self, b, coords):
        n_samples = b.shape[0]
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        for i in range(3):
            for j in range(n_samples):
                v_min_max = np.max(np.abs(b[j, :, :]))
                # extent = [coords[j, 0, 0, 2], coords[j, -1, -1, 2],
                #           coords[j, 0, 0, 1], coords[j, -1, -1, 1]]
                # extent = np.rad2deg(extent)
                extent = None
                height = coords[j, :, :, 0].mean()
                im = axs[i, j].imshow(b[j, :, :, i], cmap='gray', vmin=-v_min_max, vmax=v_min_max,
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
        n_samples = j.shape[0]
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
        for i in range(n_samples):
            # extent = [coords[i, 0, 0, 2], coords[i, -1, -1, 2],
            #           coords[i, 0, 0, 1], coords[i, -1, -1, 1]]
            # extent = np.rad2deg(extent)
            extent = None
            height = coords[i, :, :, 0].mean()
            im = axs[i].imshow(j[i, :, :], cmap='viridis', origin='lower', norm=LogNorm(), extent=extent)
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
        j = np.sum(j, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(j, cmap='viridis', origin='lower', norm=LogNorm(), extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Current density": fig})
        plt.close('all')


class SlicesCallback(Callback):

    def __init__(self, name, cube_shape, gauss_per_dB, Mm_per_ds):
        self.name = name
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.name not in pl_module.validation_outputs:
            return
        outputs = pl_module.validation_outputs[self.name]
        b = outputs['b'] * self.gauss_per_dB
        j = outputs['j'] * self.gauss_per_dB / self.Mm_per_ds
        coords = outputs['coords']

        b_cube = b.reshape([*self.cube_shape, 3]).cpu().numpy()
        j_cube = j.reshape([*self.cube_shape, 3]).cpu().numpy()
        c_cube = coords.reshape([*self.cube_shape, 3]).cpu().numpy()

        self.plot_b(b_cube, c_cube)
        self.plot_current(j_cube, c_cube)
        if 'p' in outputs:
            p = outputs['p'] * self.gauss_per_dB ** 2
            p = p.reshape([*self.cube_shape]).cpu().numpy()
            grad_P = outputs['grad_P'] * self.gauss_per_dB ** 2 / self.Mm_per_ds
            grad_P = grad_P.reshape([*self.cube_shape, 3]).cpu().numpy()
            self.plot_pressure(p, grad_P, c_cube)

        if 'rho' in outputs:
            rho = outputs['rho']
            rho = rho.reshape([*self.cube_shape]).cpu().numpy()
            self.plot_density(rho, c_cube)

    def plot_b(self, b, coords):
        n_samples = b.shape[2]
        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
        for i in range(3):
            for j in range(n_samples):
                v_min_max = np.max(np.abs(b[:, :, j]))
                extent = [coords[0, 0, j, 0], coords[-1, -1, j, 0],
                          coords[0, 0, j, 1], coords[-1, -1, j, 1]]
                extent = np.array(extent) * self.Mm_per_ds
                height = coords[:, :, j, 2].mean() * self.Mm_per_ds
                im = axs[i, j].imshow(b[:, :, j, i].T, cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                                      origin='lower', extent=extent)
                axs[i, j].set_xlabel('X [Mm]')
                axs[i, j].set_ylabel('Y [Mm]')
                # add locatable colorbar
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='B [G]')
                axs[i, j].set_title(f'{height:.02f} - $B_{["x", "y", "z"][i]}$')
        fig.tight_layout()
        wandb.log({f"{self.name} - B": fig})
        plt.close('all')

    def plot_current(self, j, coords):
        j = (j ** 2).sum(-1) ** 0.5
        n_samples = j.shape[2]
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
        for i in range(n_samples):
            extent = [coords[0, 0, i, 0], coords[-1, -1, i, 0],
                      coords[0, 0, i, 1], coords[-1, -1, i, 1]]
            extent = np.array(extent) * self.Mm_per_ds
            height = coords[:, :, i, 2].mean() * self.Mm_per_ds
            norm = LogNorm(vmin=max(j[:, :, i].min(), 1e-6), vmax=max(j[:, :, i].max(), 1e-6))
            im = axs[i].imshow(j[:, :, i].T, cmap='plasma', origin='lower', norm=norm, extent=extent)
            axs[i].set_xlabel('X [Mm]')
            axs[i].set_ylabel('Y [Mm]')
            # add locatable colorbar
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='J [G/ds]')
            axs[i].set_title(f'{height:.02f} - $|J|$')
            axs[i].set_xlabel('X [Mm]')
            axs[i].set_ylabel('Y [Mm]')
        fig.tight_layout()
        wandb.log({f"{self.name} - Current density": fig})
        plt.close('all')
        # plot integrated current density
        j = np.sum(j, axis=2)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        norm = LogNorm(vmin=max(j.min(), 1e-6), vmax=max(j.max(), 1e-6))
        im = ax.imshow(j.T, cmap='plasma', origin='lower', norm=norm, extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Current density": fig})
        plt.close('all')

    def plot_pressure(self, p, grad_P, coords):
        grad_P = np.linalg.norm(grad_P, axis=-1)
        n_samples = p.shape[2]
        fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 4, 4))
        for i in range(n_samples):
            extent = [coords[0, 0, i, 0], coords[-1, -1, i, 0],
                      coords[0, 0, i, 1], coords[-1, -1, i, 1]]
            extent = np.array(extent) * self.Mm_per_ds
            height = coords[:, :, i, 2].mean() * self.Mm_per_ds
            p_slice = p[:, :, i] + 1e-6
            im = axs[0, i].imshow(p_slice.T, cmap='viridis', origin='lower', norm=LogNorm(), extent=extent)
            # add locatable colorbar
            divider = make_axes_locatable(axs[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='P [erg/cm^3]')
            # plot grad P
            grad_P_slice = grad_P[:, :, i] + 1e-6
            im = axs[1, i].imshow(grad_P_slice.T, cmap='viridis', origin='lower', norm=LogNorm(), extent=extent)
            divider = make_axes_locatable(axs[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='$|\\nabla P|$ [erg/cm^3/Mm]')
            #
            axs[0, i].set_title(f'{height:.02f}')
            axs[1, i].set_ylabel('Y [Mm]')
        axs[0, 0].set_xlabel('X [Mm]')
        axs[1, 0].set_xlabel('X [Mm]')
        fig.tight_layout()
        wandb.log({f"{self.name} - Plasma Pressure": fig})
        plt.close('all')
        # plot integrated plasma pressure
        p = np.sum(p, axis=2)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(p.T, cmap='viridis', origin='lower', norm=LogNorm(), extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='P')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Plasma Pressure": fig})
        plt.close('all')

    def plot_density(self, rho, coords):
        n_samples = rho.shape[2]
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
        for i in range(n_samples):
            extent = [coords[0, 0, i, 0], coords[-1, -1, i, 0],
                      coords[0, 0, i, 1], coords[-1, -1, i, 1]]
            extent = np.array(extent) * self.Mm_per_ds
            height = coords[:, :, i, 2].mean() * self.Mm_per_ds
            im = axs[i].imshow(rho[:, :, i].T, cmap='cividis', origin='lower', norm=LogNorm(), extent=extent)
            axs[i].set_xlabel('X [Mm]')
            axs[i].set_ylabel('Y [Mm]')
            # add locatable colorbar
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='Density [g/cm^3]')
            axs[i].set_title(f'{height:.02f} - $\\rho$')
            axs[i].set_xlabel('X [Mm]')
            axs[i].set_ylabel('Y [Mm]')
        fig.tight_layout()
        wandb.log({f"{self.name} - Density": fig})
        plt.close('all')
        # plot integrated plasma density
        rho = np.sum(rho, axis=2)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(rho.T, cmap='cividis', origin='lower', norm=LogNorm(), extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Density [g/cm^3]')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Density": fig})
        plt.close('all')


class BoundaryCallback(Callback):

    def __init__(self, validation_dataset_key, cube_shape, gauss_per_dB, Mm_per_ds):
        self.validation_dataset_key = validation_dataset_key
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.validation_dataset_key not in pl_module.validation_outputs:
            return

        outputs = pl_module.validation_outputs[self.validation_dataset_key]
        b = outputs['b']
        b_true = outputs['b_true']

        # apply transforms
        if 'transform' in outputs:
            transform = outputs['transform']
            b = torch.einsum('ijk,ik->ij', transform, b)

        b = b * self.gauss_per_dB
        b_true = b_true * self.gauss_per_dB

        # compute diff
        b_diff = torch.abs(b - b_true)
        b_diff = torch.nanmean(b_diff.pow(2).sum(-1).pow(0.5))
        evaluation = {'b_diff': b_diff.detach()}

        # compute diff error
        if 'b_err' in outputs:
            b_err = outputs['b_err'] * self.gauss_per_dB
            b_diff_err = torch.clip(torch.abs(b - b_true) - b_err, 0)
            b_diff_err = torch.nanmean(b_diff_err.pow(2).sum(-1).pow(0.5))
            evaluation['b_diff_err'] = b_diff_err.detach()

        wandb.log({"valid": {self.validation_dataset_key: evaluation}})

        b = b.cpu().numpy().reshape([*self.cube_shape, 3])
        b_true = b_true.cpu().numpy().reshape([*self.cube_shape, 3])

        if 'original_coords' in outputs:
            original_coords = outputs['original_coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            transformed_coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            self.plot_b_coords(b, b_true, original_coords, transformed_coords)
        else:
            original_coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            self.plot_b(b, b_true)

    def plot_b(self, b, b_true):
        extent = None

        fig, axs = plt.subplots(3, 2, figsize=(8, 8))

        b_norm = np.nanmax(np.abs(b_true))
        b_norm = min(500, b_norm)

        ax = axs[0, 0]
        im = ax.imshow(b[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_x$')

        ax = axs[0, 1]
        im = ax.imshow(b_true[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_x$')

        ax = axs[1, 0]
        im = ax.imshow(b[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_y$')

        ax = axs[1, 1]
        im = ax.imshow(b_true[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_y$')

        ax = axs[2, 0]
        im = ax.imshow(b[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_z$')

        ax = axs[2, 1]
        im = ax.imshow(b_true[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_z$')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')

    def plot_b_coords(self, b, b_true, original_coords, transformed_coords):
        extent = [original_coords[..., 0].min(), original_coords[..., 0].max(),
                  original_coords[..., 1].min(), original_coords[..., 1].max()]

        fig, axs = plt.subplots(4, 2, figsize=(8, 8))

        b_norm = np.nanmax(np.abs(b_true))
        b_norm = min(500, b_norm)

        ax = axs[0, 0]
        im = ax.imshow(b[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_x$')

        ax = axs[0, 1]
        im = ax.imshow(b_true[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_x$')

        ax = axs[1, 0]
        im = ax.imshow(b[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_y$')

        ax = axs[1, 1]
        im = ax.imshow(b_true[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_y$')

        ax = axs[2, 0]
        im = ax.imshow(b[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_z$')

        ax = axs[2, 1]
        im = ax.imshow(b_true[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_z$')

        ax = axs[3, 0]
        im = ax.imshow(transformed_coords[..., 2].T, cmap='inferno', origin='lower', vmin=0, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Z [Mm]')
        ax.set_title('Transformed z')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[3, 1]
        im = ax.imshow(original_coords[..., 2].T, cmap='inferno', origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Z [Mm]')
        ax.set_title('Original z')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')


class PressureBoundaryCallback(Callback):

    def __init__(self, validation_dataset_key, cube_shape, gauss_per_dB, Mm_per_ds):
        self.validation_dataset_key = validation_dataset_key
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.validation_dataset_key not in pl_module.validation_outputs:
            return

        outputs = pl_module.validation_outputs[self.validation_dataset_key]
        p = outputs['p']
        p_true = outputs['p_true']

        p = p * self.gauss_per_dB ** 2
        p_true = p_true * self.gauss_per_dB ** 2

        # compute diff
        p_diff = torch.abs(p - p_true)
        p_diff = torch.nanmean(p_diff.pow(2).sum(-1).pow(0.5))
        evaluation = {'p_diff': p_diff.detach()}

        wandb.log({"valid": {self.validation_dataset_key: evaluation}})

        p = p.cpu().numpy().reshape(self.cube_shape)
        p_true = p_true.cpu().numpy().reshape(self.cube_shape)

        coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
        self.plot_p(p, p_true, coords)

    def plot_p(self, p, p_true, coords):
        extent = None
        p_norm = LogNorm(vmin=np.nanmin(p_true), vmax=np.nanmax(p_true))

        fig, axs = plt.subplots(1, 2, figsize=(8, 8))

        ax = axs[0]
        im = ax.imshow(p.T, cmap='viridis', norm=p_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[erg/cm^3]')
        ax.set_title('Predicted Pressure')

        ax = axs[1]
        im = ax.imshow(p_true.T, cmap='viridis', norm=p_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[erg/cm^3]')
        ax.set_title('True Pressure')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')


class DisambiguationCallback(Callback):

    def __init__(self, validation_dataset_key, cube_shape, gauss_per_dB, Mm_per_ds, name=None):
        self.validation_dataset_key = validation_dataset_key
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds
        self.name = name

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.validation_dataset_key not in pl_module.validation_outputs:
            return

        outputs = pl_module.validation_outputs[self.validation_dataset_key]
        flip = outputs['flip']
        b_true = outputs['b_true']

        # load predicted los_trv_azi
        b_xyz = outputs['b']
        if 'transform' in outputs:
            transform = outputs['transform']
            b_xyz = torch.einsum('ijk,ik->ij', transform, b_xyz)
        b_xyz = b_xyz * self.gauss_per_dB
        b_pred = img_to_los_trv_azi(b_xyz, f=torch)

        flip = flip.cpu().numpy().reshape([*self.cube_shape])
        azimuth_true = b_true[..., 2].cpu().numpy().reshape([*self.cube_shape]) % (2 * np.pi)
        azimuth_amb = azimuth_true % np.pi

        azimuth = (azimuth_amb + np.round(flip) * np.pi) % (2 * np.pi)
        azimuth_pred = b_pred[..., 2].cpu().numpy().reshape([*self.cube_shape]) % (2 * np.pi)

        extent = None

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        ax = axs[0, 0]
        im = ax.imshow(azimuth_true.T, cmap='twilight', origin='lower',
                       extent=extent, vmin=0, vmax=2 * np.pi)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Ambiguous Azimuth [rad]')

        ax = axs[0, 1]
        im = ax.imshow(azimuth.T, cmap='twilight', origin='lower',
                       extent=extent, vmin=0, vmax=2 * np.pi)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Disambiguated Azimuth [rad]')

        ax = axs[1, 0]
        im = ax.imshow(azimuth_pred.T, cmap='twilight', origin='lower',
                       extent=extent, vmin=0, vmax=2 * np.pi)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Predicted Azimuth [rad]')

        ax = axs[1, 1]
        im = ax.imshow(flip.T, cmap='bwr', origin='lower', extent=extent, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Flip Probability')

        axs[0, 0].set_title('Input Azimuth')
        axs[0, 1].set_title('Disambiguated Azimuth')
        axs[1, 0].set_title('Predicted Azimuth')
        axs[1, 1].set_title('Flip Probability')

        fig.tight_layout()
        name = f"{self.validation_dataset_key} - Disambiguation" if self.name is None else self.name
        wandb.log({name: fig})
        plt.close('all')


class LosTrvAziBoundaryCallback(Callback):

    def __init__(self, validation_dataset_key, cube_shape, gauss_per_dB, Mm_per_ds):
        self.validation_dataset_key = validation_dataset_key
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.validation_dataset_key not in pl_module.validation_outputs:
            return

        outputs = pl_module.validation_outputs[self.validation_dataset_key]
        b_xyz = outputs['b']
        b_true = outputs['b_true']

        # apply transforms
        if 'transform' in outputs:
            transform = outputs['transform']
            b_xyz = torch.einsum('ijk,ik->ij', transform, b_xyz)

        b_xyz = b_xyz * self.gauss_per_dB
        b_los_trv_azi = img_to_los_trv_azi(b_xyz, f=torch)
        b_true[..., 0:2] = b_true[..., 0:2] * self.gauss_per_dB

        b_true_xyz = los_trv_azi_to_img(b_true, f=torch)
        # compute diff
        b_diff = b_xyz - b_true_xyz
        b_diff = torch.nanmean(b_diff.pow(2).sum(-1).pow(0.5))
        evaluation = {'b_diff': b_diff.detach()}
        #
        b_xyz_amb = los_trv_azi_to_img(b_los_trv_azi, ambiguous=True, f=torch)
        b_true_xyz_amb = los_trv_azi_to_img(b_true, ambiguous=True, f=torch)
        # compute diff
        b_diff = torch.abs(b_xyz_amb - b_true_xyz_amb)
        b_diff = torch.nanmean(b_diff.pow(2).sum(-1).pow(0.5))
        evaluation['b_amb_diff'] = b_diff.detach()

        wandb.log({"valid": {self.validation_dataset_key: evaluation}})

        b_los_trv_azi = b_los_trv_azi.cpu().numpy().reshape([*self.cube_shape, 3])
        b_true = b_true.cpu().numpy().reshape([*self.cube_shape, 3])
        b_xyz = b_xyz.cpu().numpy().reshape([*self.cube_shape, 3])
        b_true_xyz = b_true_xyz.cpu().numpy().reshape([*self.cube_shape, 3])

        if 'original_coords' in outputs:
            original_coords = outputs['original_coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            transformed_coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            self.plot_b_coords(b_los_trv_azi, b_true, original_coords, transformed_coords)
        else:
            original_coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            self.plot_b(b_los_trv_azi, b_true, original_coords)

        self.plot_bxyz(b_xyz, b_true_xyz)

    def plot_bxyz(self, b, b_true):
        extent = None

        fig, axs = plt.subplots(3, 2, figsize=(8, 8))

        b_norm = np.nanmax(np.abs(b_true))
        b_norm = min(500, b_norm)

        ax = axs[0, 0]
        im = ax.imshow(b[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_x$')

        ax = axs[0, 1]
        im = ax.imshow(b_true[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_x$')

        ax = axs[1, 0]
        im = ax.imshow(b[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_y$')

        ax = axs[1, 1]
        im = ax.imshow(b_true[..., 1].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_y$')

        ax = axs[2, 0]
        im = ax.imshow(b[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('Predicted $B_z$')

        ax = axs[2, 1]
        im = ax.imshow(b_true[..., 2].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('True $B_z$')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - Bxyz": fig})
        plt.close('all')

    def plot_b(self, b, b_true, original_coords):
        extent = [original_coords[..., 0].min(), original_coords[..., 0].max(),
                  original_coords[..., 1].min(), original_coords[..., 1].max()]

        fig, axs = plt.subplots(3, 2, figsize=(8, 8))

        b_norm = np.nanmax(np.abs(b_true[..., 0]))
        b_norm = min(500, b_norm)

        ax = axs[0, 0]
        im = ax.imshow(b[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('B_los')

        ax = axs[0, 1]
        im = ax.imshow(b_true[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('B_los_true')

        ax = axs[1, 0]
        im = ax.imshow(b[..., 1].T, cmap='gray', vmin=0, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('B_trv')

        ax = axs[1, 1]
        im = ax.imshow(b_true[..., 1].T, cmap='gray', vmin=0, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[G]')
        ax.set_title('B_trv_true')

        ax = axs[2, 0]
        im = ax.imshow(b[..., 2].T % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower',
                       extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[deg]')
        ax.set_title('B_azi')

        ax = axs[2, 1]
        im = ax.imshow(b_true[..., 2].T % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower',
                       extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='[deg]')
        ax.set_title('B_azi_true')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')

    def plot_b_coords(self, b, b_true, original_coords, transformed_coords):
        extent = [original_coords[..., 0].min(), original_coords[..., 0].max(),
                  original_coords[..., 1].min(), original_coords[..., 1].max()]

        fig, axs = plt.subplots(4, 2, figsize=(8, 8))

        b_norm = np.nanmax(np.abs(b_true[..., 0]))
        b_norm = min(500, b_norm)

        ax = axs[0, 0]
        im = ax.imshow(b[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B$_{los}$ [G]')
        ax.set_title(r'$B_\text{LOS}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[0, 1]
        im = ax.imshow(b_true[..., 0].T, cmap='gray', vmin=-b_norm, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B$_{los,true}$ [G]')
        ax.set_title(r'$B_\text{LOS,true}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[1, 0]
        im = ax.imshow(b[..., 1].T, cmap='gray', vmin=0, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B$_{trv}$ [G]')
        ax.set_title(r'$B_\text{TRV}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[1, 1]
        im = ax.imshow(b_true[..., 1].T, cmap='gray', vmin=0, vmax=b_norm, origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='B$_{trv,true}$ [G]')
        ax.set_title(r'$B_\text{TRV,true}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[2, 0]
        im = ax.imshow(b[..., 2].T % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower',
                       extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Azimuth [rad]')
        ax.set_title(r'$B_\text{AZI}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[2, 1]
        im = ax.imshow(b_true[..., 2].T % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower',
                       extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Azimuth [rad]')
        ax.set_title(r'$B_\text{AZI,true}$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[3, 0]
        im = ax.imshow(transformed_coords[..., 2].T, cmap='inferno', origin='lower', vmin=0, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Z [Mm]')
        ax.set_title('Transformed $z$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = axs[3, 1]
        im = ax.imshow(original_coords[..., 2].T, cmap='inferno', origin='lower', extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Z [Mm]')
        ax.set_title('Original $z$')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        fig.tight_layout()
        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')


class MetricsCallback(Callback):

    def __init__(self, validation_dataset_key, gauss_per_dB, Mm_per_ds):
        self.validation_dataset_key = validation_dataset_key
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.validation_dataset_key not in pl_module.validation_outputs:
            return
        outputs = pl_module.validation_outputs[self.validation_dataset_key]

        b = outputs['b'] * self.gauss_per_dB
        j = outputs['j'] * self.gauss_per_dB / self.Mm_per_ds

        div = outputs['div'] * self.gauss_per_dB / self.Mm_per_ds

        norm = torch.norm(b, dim=-1) * torch.norm(j, dim=-1)
        sigma = torch.norm(torch.cross(j, b, dim=-1), dim=-1) / norm
        j_weight = torch.norm(j, dim=-1)
        angle = (sigma * j_weight).sum() / j_weight.sum()
        angle = torch.clip(angle, -1. + 1e-7, 1. - 1e-7)
        theta_J = torch.arcsin(angle)
        theta_J = torch.rad2deg(theta_J)

        sigma_J = (torch.norm(torch.cross(j, b, dim=-1), dim=-1) / (torch.norm(b, dim=-1) + 1e-7)).sum() / (
                torch.norm(j, dim=-1).sum() + 1e-7)

        b_norm = b.pow(2).sum(-1).pow(0.5) + 1e-7
        div_loss = (div / b_norm).mean()

        ff_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1).pow(0.5) / b_norm
        ff_loss = ff_loss.mean()

        wandb.log({"valid": {"divergence": div_loss.cpu().numpy(),
                             "force-free": ff_loss.cpu().numpy(),
                             "sigma_J": sigma_J.cpu().numpy(),
                             "theta_J": theta_J.cpu().numpy()}})


class AdvanceDatamoduleStep(Callback):
    def __init__(self, data_module, every_n):
        super().__init__()
        self.every_n = every_n
        self.data_module = data_module
        # assures that train epoch start is called at least once before we start advancing steps
        # avoids errors for continued training from an interrupted checkpoint
        self.initialized = False

    def on_train_epoch_start(self, trainer, pl_module):
        self._print_step()
        self.initialized = True

    @rank_zero_only
    def _print_step(self):
        data_module = self.data_module
        print(f'\nStep {data_module.step + 1:03d}/{len(data_module.train_configs):03d}; ID: {data_module.current_id}')

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.initialized:
            return
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.every_n != 0:
            return

        data_module = self.data_module
        data_module.step += 1

        # if we've used all configs, stop training cleanly
        if data_module.step >= data_module.total_steps:
            print('All training files processed. Stopping training...')
            trainer.should_stop = True
