import numpy as np
import torch
import wandb
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only

from nf2.data.util import cartesian_to_spherical, vector_cartesian_to_spherical, img_to_los_trv_azi, los_trv_azi_to_img


def _log_norm(values, floor=1e-6):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return LogNorm(vmin=floor, vmax=floor * 10)
    vmin = max(values.min(), floor)
    vmax = max(values.max(), vmin * 10)
    return LogNorm(vmin=vmin, vmax=vmax)


def _trapezoid(y, x, axis):
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return integrate(y, x=x, axis=axis)


def _cartesian_extent(coords):
    return [np.nanmin(coords[..., 0]), np.nanmax(coords[..., 0]),
            np.nanmin(coords[..., 1]), np.nanmax(coords[..., 1])]


def _cartesian_extent_title(coords):
    x_min, x_max, y_min, y_max = _cartesian_extent(coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return (f'X: {x_min:.2f} to {x_max:.2f} Mm, '
            f'Y: {y_min:.2f} to {y_max:.2f} Mm; '
            f'center: ({x_center:.2f}, {y_center:.2f}) Mm')


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
        fig, plot_axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12), squeeze=False)
        for i in range(3):
            for j in range(n_samples):
                height = coords[j, :, :, 0].mean()
                b_slice = b[j, :, :, i]
                v_min_max = np.nanmax(np.abs(b_slice))
                v_min_max = max(v_min_max, 1)
                im = plot_axs[i, j].imshow(b_slice, cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                                            origin='lower', extent=None)
                divider = make_axes_locatable(plot_axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='B [G]')
                if i == 2:
                    plot_axs[i, j].set_xlabel('Longitude [deg]')
                else:
                    plot_axs[i, j].tick_params(labelbottom=False)
                if j == 0:
                    plot_axs[i, j].set_ylabel('Latitude [deg]')
                else:
                    plot_axs[i, j].tick_params(labelleft=False)
                plot_axs[i, j].set_title(f'{height:.02f} - $B_{["r", "t", "p"][i]}$')
        fig.tight_layout()
        wandb.log({f"{self.name} - B": fig})
        plt.close('all')

    def plot_current(self, j, coords):
        j = (j ** 2).sum(-1) ** 0.5
        n_samples = j.shape[0]
        width_ratios = [1] * n_samples + [0.05]
        fig, axs = plt.subplots(1, n_samples + 1, figsize=(n_samples * 4 + 0.4, 4),
                                gridspec_kw={'width_ratios': width_ratios}, constrained_layout=True,
                                squeeze=False)
        plot_axs = axs[0, :-1]
        cbar_ax = axs[0, -1]
        norm = _log_norm(j)
        for i in range(n_samples):
            height = coords[i, :, :, 0].mean()
            im = self._plot_spherical_map(plot_axs[i], j[i, :, :], coords[i], cmap='plasma', norm=norm)
            plot_axs[i].set_xlabel('Longitude [deg]')
            if i == 0:
                plot_axs[i].set_ylabel('Latitude [deg]')
            else:
                plot_axs[i].tick_params(labelleft=False)
            plot_axs[i].set_title(f'{height:.02f} $R_\\odot$ - $|J|$')
        fig.colorbar(im, cax=cbar_ax, label=r'$|J|$ [G/Mm]')
        wandb.log({f"{self.name} - Current density": fig})
        plt.close('all')
        # plot integrated current density
        j, j_label = self._integrate_radial_current(j, coords)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = self._plot_spherical_map(ax, j, coords[0], cmap='plasma', norm=_log_norm(j), physical_units=True)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=j_label)
        ax.set_xlabel('Longitudinal arc length [Mm]')
        ax.set_ylabel('Latitudinal arc length [Mm]')
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Current density": fig})
        plt.close('all')

    @staticmethod
    def _plot_spherical_map(ax, values, coords, physical_units=False, **kwargs):
        longitude_rad = np.unwrap(coords[..., 2], axis=1)
        longitude = np.rad2deg(longitude_rad)
        latitude_rad = np.pi / 2 - coords[..., 1]
        latitude = np.rad2deg(latitude_rad)

        if physical_units:
            radius_Mm = np.nanmean(coords[..., 0]) * (1 * u.solRad).to_value(u.Mm)
            longitude = radius_Mm * (longitude_rad - np.nanmean(longitude_rad))
            latitude = radius_Mm * latitude_rad
            aspect = 'equal'
        else:
            aspect = 'auto'

        extent = [np.nanmin(longitude), np.nanmax(longitude),
                  np.nanmin(latitude), np.nanmax(latitude)]

        im = ax.imshow(np.flipud(values), origin='lower', extent=extent, aspect=aspect, **kwargs)
        return im

    def _integrate_radial_current(self, j, coords):
        if j.shape[0] < 2:
            return j[0], r'$|J|$ [G/Mm]'

        radius_Mm = coords[..., 0] * (1 * u.solRad).to_value(u.Mm)
        integrated_j = _trapezoid(j, x=radius_Mm, axis=0)
        return integrated_j, r'$\int |J|\,dr$ [G]'


class SphericalFITSComparisonCallback(Callback):

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

        b_cube = b.reshape([*self.cube_shape, 3]).cpu().numpy()
        j_cube = j.reshape([*self.cube_shape, 3]).cpu().numpy()
        if 'spherical_coords' in outputs:
            spherical_coords = outputs['spherical_coords'].reshape([*self.cube_shape, 3]).cpu().numpy()
        else:
            coords = outputs['coords'].reshape([*self.cube_shape, 3]).cpu().numpy()
            coords = coords * self.Mm_per_ds / (1 * u.solRad).to_value(u.Mm)
            spherical_coords = cartesian_to_spherical(coords)

        model_br = vector_cartesian_to_spherical(b_cube, spherical_coords)[0, ..., 0]
        reference_br = outputs['reference_br'].reshape(self.cube_shape).cpu().numpy()[0]
        reference_lon = outputs['reference_lon'].reshape(self.cube_shape).cpu().numpy()[0]
        reference_lat = outputs['reference_lat'].reshape(self.cube_shape).cpu().numpy()[0]

        j_cube = np.linalg.norm(j_cube, axis=-1)
        integrated_current, current_label = self._integrate_radial_current(j_cube, spherical_coords)

        self.plot_comparison(reference_br, model_br, integrated_current,
                             reference_lon, reference_lat, current_label)

    def plot_comparison(self, reference_br, model_br, integrated_current, longitude, latitude, current_label):
        extent = [np.nanmin(longitude), np.nanmax(longitude),
                  np.nanmin(latitude), np.nanmax(latitude)]
        br_norm_value = np.nanmax(np.abs([reference_br, model_br]))
        br_norm_value = max(min(br_norm_value, 1000), 1)
        br_norm = Normalize(vmin=-br_norm_value, vmax=br_norm_value)
        current_norm = _log_norm(integrated_current)

        fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)

        im = axs[0].imshow(reference_br, origin='lower', extent=extent, cmap='gray', norm=br_norm)
        axs[0].set_title('Reference Br')
        self._add_colorbar(fig, axs[0], im, 'Br [G]')

        im = axs[1].imshow(model_br, origin='lower', extent=extent, cmap='gray', norm=br_norm)
        axs[1].set_title('Model Br')
        self._add_colorbar(fig, axs[1], im, 'Br [G]')

        im = axs[2].imshow(integrated_current, origin='lower', extent=extent, cmap='inferno', norm=current_norm)
        axs[2].set_title('Integrated Current Density')
        self._add_colorbar(fig, axs[2], im, current_label)

        for i, ax in enumerate(axs):
            ax.set_xlabel('Carrington Longitude [deg]')
            if i == 0:
                ax.set_ylabel('Carrington Latitude [deg]')
            else:
                ax.tick_params(labelleft=False)

        fig.tight_layout()
        wandb.log({f"{self.name} - FITS Comparison": fig})
        plt.close('all')

    @staticmethod
    def _add_colorbar(fig, ax, im, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label=label)

    def _integrate_radial_current(self, j, spherical_coords):
        if j.shape[0] < 2:
            return j[0], r'$|J|$ [G/Mm]'

        radius_Mm = spherical_coords[..., 0] * (1 * u.solRad).to_value(u.Mm)
        integrated_j = _trapezoid(j, x=radius_Mm, axis=0)
        return integrated_j, r'$\int |J|\,dr$ [G]'


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

    def plot_b(self, b, coords):
        n_samples = b.shape[2]
        fig, plot_axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12), squeeze=False)
        for i in range(3):
            for j in range(n_samples):
                extent = [coords[0, 0, j, 0], coords[-1, -1, j, 0],
                          coords[0, 0, j, 1], coords[-1, -1, j, 1]]
                extent = np.array(extent) * self.Mm_per_ds
                height = coords[:, :, j, 2].mean() * self.Mm_per_ds
                b_slice = b[:, :, j, i]
                v_min_max = np.nanmax(np.abs(b_slice))
                v_min_max = max(v_min_max, 1)
                ax = plot_axs[i, j]
                im = ax.imshow(b_slice.T, cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                               origin='lower', extent=extent)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='B [G]')
                if i == 2:
                    ax.set_xlabel('X [Mm]')
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel('Y [Mm]')
                else:
                    ax.tick_params(labelleft=False)
                ax.set_title(f'{height:.02f} - $B_{["x", "y", "z"][i]}$')
        fig.tight_layout()
        wandb.log({f"{self.name} - B": fig})
        plt.close('all')

    def plot_current(self, j, coords):
        j = (j ** 2).sum(-1) ** 0.5
        n_samples = j.shape[2]
        width_ratios = [1] * n_samples + [0.05]
        fig, axs = plt.subplots(1, n_samples + 1, figsize=(n_samples * 4 + 0.4, 4),
                                gridspec_kw={'width_ratios': width_ratios}, constrained_layout=True,
                                squeeze=False)
        plot_axs = axs[0, :-1]
        cbar_ax = axs[0, -1]
        norm = _log_norm(j)
        for i in range(n_samples):
            extent = [coords[0, 0, i, 0], coords[-1, -1, i, 0],
                      coords[0, 0, i, 1], coords[-1, -1, i, 1]]
            extent = np.array(extent) * self.Mm_per_ds
            height = coords[:, :, i, 2].mean() * self.Mm_per_ds
            im = plot_axs[i].imshow(j[:, :, i].T, cmap='plasma', origin='lower', norm=norm, extent=extent)
            plot_axs[i].set_xlabel('X [Mm]')
            if i == 0:
                plot_axs[i].set_ylabel('Y [Mm]')
            else:
                plot_axs[i].tick_params(labelleft=False)
            plot_axs[i].set_title(f'{height:.02f} - $|J|$')
        fig.colorbar(im, cax=cbar_ax, label=r'$|J|$ [G/Mm]')
        wandb.log({f"{self.name} - Current density": fig})
        plt.close('all')
        # plot integrated current density
        j, j_label = self._integrate_height_current(j, coords)
        extent = [coords[0, 0, 0, 0], coords[-1, -1, 0, 0],
                  coords[0, 0, 0, 1], coords[-1, -1, 0, 1]]
        extent = np.array(extent) * self.Mm_per_ds
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        norm = _log_norm(j)
        im = ax.imshow(j.T, cmap='plasma', origin='lower', norm=norm, extent=extent)
        # add locatable colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=j_label)
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        #
        fig.tight_layout()
        wandb.log({f"{self.name} - Integrated Current density": fig})
        plt.close('all')

    def _integrate_height_current(self, j, coords):
        if j.shape[2] < 2:
            return j[:, :, 0], r'$|J|$ [G/Mm]'

        height_Mm = coords[..., 2] * self.Mm_per_ds
        integrated_j = _trapezoid(j, x=height_Mm, axis=2)
        return integrated_j, r'$\int |J|\,dz$ [G]'

class BoundaryCallback(Callback):

    def __init__(self, validation_dataset_key, cube_shape, gauss_per_dB, Mm_per_ds, component_labels=None, **kwargs):
        self.validation_dataset_key = validation_dataset_key
        self.cube_shape = cube_shape
        self.gauss_per_dB = gauss_per_dB
        self.Mm_per_ds = Mm_per_ds
        if component_labels is None and validation_dataset_key == 'synoptic_valid':
            component_labels = ['r', 't', 'p']
        self.component_labels = component_labels or ['x', 'y', 'z']

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
            coords = outputs['coords'].cpu().numpy().reshape([*self.cube_shape, 3]) * self.Mm_per_ds
            self.plot_b(b, b_true, coords)

    def plot_b(self, b, b_true, coords=None):
        extent = _cartesian_extent(coords) if coords is not None else None

        fig, axs = plt.subplots(3, 3, figsize=(8.4, 8),
                                gridspec_kw={'width_ratios': [1, 1, 0.05]}, constrained_layout=True)
        plot_axs = axs[:, :2]
        cbar_axs = axs[:, 2]

        b_norm = np.nanmax(np.abs(b_true))
        b_norm = min(500, b_norm)

        for i, label in enumerate(self.component_labels):
            im = plot_axs[i, 0].imshow(b[..., i].T, cmap='gray', vmin=-b_norm, vmax=b_norm,
                                       origin='lower', extent=extent)
            plot_axs[i, 0].set_title(f'Predicted $B_{label}$')
            im = plot_axs[i, 1].imshow(b_true[..., i].T, cmap='gray', vmin=-b_norm, vmax=b_norm,
                                       origin='lower', extent=extent)
            plot_axs[i, 1].set_title(f'True $B_{label}$')
            fig.colorbar(im, cax=cbar_axs[i], label='[G]')
            if coords is not None:
                for ax in plot_axs[i]:
                    ax.set_xlabel('X [Mm]')
                    ax.set_ylabel('Y [Mm]')

        if coords is not None:
            fig.suptitle(_cartesian_extent_title(coords))

        wandb.log({f"{self.validation_dataset_key} - B": fig})
        plt.close('all')

    def plot_b_coords(self, b, b_true, original_coords, transformed_coords):
        extent = _cartesian_extent(original_coords)

        fig, axs = plt.subplots(4, 3, figsize=(8.4, 8),
                                gridspec_kw={'width_ratios': [1, 1, 0.05]}, constrained_layout=True)
        plot_axs = axs[:, :2]
        cbar_axs = axs[:, 2]

        b_norm = np.nanmax(np.abs(b_true))
        b_norm = min(500, b_norm)

        for i, label in enumerate(self.component_labels):
            im = plot_axs[i, 0].imshow(b[..., i].T, cmap='gray', vmin=-b_norm, vmax=b_norm,
                                       origin='lower', extent=extent)
            plot_axs[i, 0].set_title(f'Predicted $B_{label}$')
            im = plot_axs[i, 1].imshow(b_true[..., i].T, cmap='gray', vmin=-b_norm, vmax=b_norm,
                                       origin='lower', extent=extent)
            plot_axs[i, 1].set_title(f'True $B_{label}$')
            fig.colorbar(im, cax=cbar_axs[i], label='[G]')
            for ax in plot_axs[i]:
                ax.set_xlabel('X [Mm]')
                ax.set_ylabel('Y [Mm]')

        ax = plot_axs[3, 0]
        im = ax.imshow(transformed_coords[..., 2].T, cmap='inferno', origin='lower', vmin=0, extent=extent)
        ax.set_title('Transformed z')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

        ax = plot_axs[3, 1]
        im = ax.imshow(original_coords[..., 2].T, cmap='inferno', origin='lower', extent=extent)
        ax.set_title('Original z')
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')
        fig.colorbar(im, cax=cbar_axs[3], label='Z [Mm]')
        fig.suptitle(_cartesian_extent_title(original_coords))

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

        norm = torch.norm(b, dim=-1) * torch.norm(j, dim=-1) + 1e-7
        sigma = torch.norm(torch.cross(j, b, dim=-1), dim=-1) / norm
        j_weight = torch.norm(j, dim=-1)
        angle = (sigma * j_weight).sum() / (j_weight.sum() + 1e-7)
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
        print(f'\nStep {data_module.step + 1:03d}/{len(data_module.boundaries):03d}; ID: {data_module.current_id}')

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
