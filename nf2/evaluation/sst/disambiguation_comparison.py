import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize


class DisambiguationComparison(ABC):
    def __init__(
            self,
            nf2_paths,
            out_path,
            output_name,
            Mm_per_pixel,
            reference_name='Reference',
            nf2_names=None,
            field_limit=1000,
            alpha_limit=1000,
            diff_limit=1000):
        self.nf2_paths = nf2_paths if isinstance(nf2_paths, (list, tuple)) else [nf2_paths]
        self.out_path = out_path
        self.output_name = output_name
        self.Mm_per_pixel = Mm_per_pixel
        self.reference_name = reference_name
        self.nf2_names = nf2_names or ['NF2'] * len(self.nf2_paths)
        self.field_limit = field_limit
        self.alpha_limit = alpha_limit
        self.diff_limit = diff_limit
        if len(self.nf2_names) != len(self.nf2_paths):
            raise ValueError('NF2 names and paths must have the same length.')

    @abstractmethod
    def load_reference_field(self):
        """Return the reference magnetic field with shape (ny, nx, 3)."""

    def load_nf2_field(self, path):
        with np.load(path) as data:
            return data['b'][:, :, 0]

    def load_nf2_fields(self):
        return [(name, self.load_nf2_field(path)) for name, path in zip(self.nf2_names, self.nf2_paths)]

    def run(self):
        os.makedirs(self.out_path, exist_ok=True)
        b_ref = self.load_reference_field()
        nf2_fields = self.load_nf2_fields()
        b_ref, nf2_fields = self.align_fields(b_ref, nf2_fields)
        self.validate_fields(b_ref, nf2_fields)
        self.plot_comparison(b_ref, nf2_fields)

    def align_fields(self, b_ref, nf2_fields):
        min_y = min([b_ref.shape[0], *[b_nf2.shape[0] for _, b_nf2 in nf2_fields]])
        min_x = min([b_ref.shape[1], *[b_nf2.shape[1] for _, b_nf2 in nf2_fields]])
        b_ref = b_ref[:min_y, :min_x, :]
        nf2_fields = [(name, b_nf2[:min_y, :min_x, :]) for name, b_nf2 in nf2_fields]
        return b_ref, nf2_fields

    def validate_fields(self, b_ref, nf2_fields):
        if b_ref.ndim != 3 or b_ref.shape[-1] != 3:
            raise ValueError(f'Expected magnetic field arrays with shape (ny, nx, 3), got {b_ref.shape}')
        for name, b_nf2 in nf2_fields:
            if b_ref.shape != b_nf2.shape:
                raise ValueError(
                    f'Reference and {name} fields must have the same shape: {b_ref.shape} != {b_nf2.shape}')

    def azimuth_flip(self, b):
        b_mag = np.linalg.norm(b, axis=-1)
        azimuth = np.arctan2(-b[..., 0], b[..., 1])
        mask = np.mod(azimuth, 2 * np.pi) > np.pi
        alpha = np.clip(b_mag / self.alpha_limit, 0, 1)
        return mask, alpha

    def plot_comparison(self, b_ref, nf2_fields):
        n_nf2 = len(nf2_fields)
        ncols = 1 + 2 * n_nf2
        field_end_col = 1 + n_nf2
        diffs = [(name, b_nf2 - b_ref) for name, b_nf2 in nf2_fields]
        vector_diffs = [(name, np.linalg.norm(b_ref - b_nf2, axis=-1)) for name, b_nf2 in nf2_fields]
        ref_mask, ref_alpha = self.azimuth_flip(b_ref)
        nf2_masks = [(name, *self.azimuth_flip(b_nf2)) for name, b_nf2 in nf2_fields]

        field_norm = Normalize(vmin=-self.field_limit, vmax=self.field_limit)
        signed_diff_norm = Normalize(vmin=-self.diff_limit, vmax=self.diff_limit)
        vector_diff_norm = Normalize(vmin=0, vmax=self.diff_limit)
        extent = [0, b_ref.shape[0] * self.Mm_per_pixel, 0, b_ref.shape[1] * self.Mm_per_pixel]

        fig = plt.figure(figsize=(3 * ncols, 10), constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=6,
            ncols=ncols,
            height_ratios=[1, 1, 1, 0.08, 1, 0.08],
        )

        shared_ax = None
        axes = []
        for i in range(3):
            row = []
            for j in range(ncols):
                ax = fig.add_subplot(gs[i, j], sharex=shared_ax, sharey=shared_ax)
                if shared_ax is None:
                    shared_ax = ax
                row.append(ax)
            axes.append(row)
        component_caxes = [fig.add_subplot(gs[3, 0:field_end_col]), fig.add_subplot(gs[3, field_end_col:ncols])]
        mask_axes = [fig.add_subplot(gs[4, j], sharex=shared_ax, sharey=shared_ax) for j in range(ncols)]
        mask_caxes = [fig.add_subplot(gs[5, 0:field_end_col]), fig.add_subplot(gs[5, field_end_col:ncols])]

        field_kwargs = {'cmap': 'gray', 'norm': field_norm, 'origin': 'lower', 'extent': extent}
        component_names = [r'B$_\mathrm{x}$', r'B$_\mathrm{y}$', r'B$_\mathrm{z}$']

        field_im = None
        signed_diff_im = None
        for i, component_name in enumerate(component_names):
            field_im = axes[i][0].imshow(b_ref[..., i].T, **field_kwargs)
            axes[i][0].set_title(f'{component_name} {self.reference_name}')

            for j, (name, b_nf2) in enumerate(nf2_fields, start=1):
                axes[i][j].imshow(b_nf2[..., i].T, **field_kwargs)
                axes[i][j].set_title(f'{component_name} {name}')

            for j, (name, b_diff) in enumerate(diffs, start=field_end_col):
                signed_diff_im = axes[i][j].imshow(
                    b_diff[..., i].T,
                    cmap='RdBu_r',
                    norm=signed_diff_norm,
                    origin='lower',
                    extent=extent,
                )
                axes[i][j].set_title(rf'$\Delta$ {component_name} {name}')

        mask_cmap = ListedColormap(['blue', 'red'])
        ref_mask_im = mask_axes[0].imshow(
            ref_mask.T,
            cmap=mask_cmap,
            vmin=0,
            vmax=1,
            alpha=ref_alpha.T,
            origin='lower',
            extent=extent,
        )
        mask_axes[0].set_title(f'Azimuth Flip {self.reference_name}')

        for j, (name, mask, alpha) in enumerate(nf2_masks, start=1):
            mask_axes[j].imshow(
                mask.T,
                cmap=mask_cmap,
                vmin=0,
                vmax=1,
                alpha=alpha.T,
                origin='lower',
                extent=extent,
            )
            mask_axes[j].set_title(f'Azimuth Flip {name}')

        vector_diff_im = None
        for j, (name, vector_diff) in enumerate(vector_diffs, start=field_end_col):
            vector_diff_im = mask_axes[j].imshow(
                vector_diff.T,
                cmap='Reds',
                norm=vector_diff_norm,
                origin='lower',
                extent=extent,
            )
            mask_axes[j].set_title(rf'|B$_\mathrm{{ref}}$ - B| {name}')

        for ax in [*sum(axes, []), *mask_axes]:
            ax.tick_params(labelbottom=False, labelleft=False)
        for ax in mask_axes:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('X [Mm]')
        for ax in [axes[0][0], axes[1][0], axes[2][0], mask_axes[0]]:
            ax.tick_params(labelleft=True)
            ax.set_ylabel('Y [Mm]')

        fig.colorbar(
            field_im,
            cax=component_caxes[0],
            orientation='horizontal',
            label='Magnetic Field Strength [G]',
        )
        fig.colorbar(
            signed_diff_im,
            cax=component_caxes[1],
            orientation='horizontal',
            label='Signed Difference [G]',
        )

        mask_cb = fig.colorbar(
            ref_mask_im,
            cax=mask_caxes[0],
            orientation='horizontal',
            ticks=[0, 1],
        )
        mask_cb.set_label('Azimuth Flip')

        fig.colorbar(
            vector_diff_im,
            cax=mask_caxes[1],
            orientation='horizontal',
            label=r'|B$_\mathrm{ref}$ - B| [G]',
        )

        fig.savefig(os.path.join(self.out_path, self.output_name), dpi=100, transparent=True)
        plt.close(fig)
