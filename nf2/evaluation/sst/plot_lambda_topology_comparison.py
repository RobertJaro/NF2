import argparse
import os
from dataclasses import dataclass

import numpy as np
from matplotlib import gridspec, pyplot as plt
from matplotlib.colors import LogNorm


DEFAULT_DATA_DIR = '/glade/work/rjarolim/nf2/topology/npzs'
USE_DUMMY_Q_T = False


@dataclass
class Dataset:
    title: str
    data: dict
    bz_boundary_map: np.ndarray | None = None
    current_map: np.ndarray | None = None
    squashing: dict | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare MURaM topology diagnostics for different lambda_ff configurations.'
    )
    parser.add_argument(
        '--output-file',
        default='/glade/work/rjarolim/nf2/topology/results/curvature/muram_mfr_lambda_topology.png',
    )
    parser.add_argument('--data-muram', default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr.npz'))
    parser.add_argument(
        '--data-1slices-ff1e-3',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-3_v01.npz'),
    )
    parser.add_argument(
        '--data-2slices-ff1e-3',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-3_v01.npz'),
    )
    parser.add_argument(
        '--data-1slices-ff5e-4',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff5.0e-4_v01.npz'),
    )
    parser.add_argument(
        '--data-2slices-ff5e-4',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-4_v01.npz'),
    )
    parser.add_argument(
        '--data-1slices-ff1e-4',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-4_v01.npz'),
    )
    parser.add_argument(
        '--data-2slices-ff1e-4',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-4_v01.npz'),
    )
    parser.add_argument(
        '--data-1slices-ff5e-5',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff5.0e-5_v01.npz'),
    )
    parser.add_argument(
        '--data-2slices-ff5e-5',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff5.0e-5_v01.npz'),
    )
    parser.add_argument(
        '--data-1slices-ff1e-5',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_1slices_ff1.0e-5_v01.npz'),
    )
    parser.add_argument(
        '--data-2slices-ff1e-5',
        default=os.path.join(DEFAULT_DATA_DIR, 'muram_mfr_2slices_ff1.0e-5_v01.npz'),
    )
    parser.add_argument('--x-slice', type=float, default=-13)
    parser.add_argument('--y-lim', type=float, nargs=2, metavar=('Y_MIN', 'Y_MAX'), default=[2, 12])
    parser.add_argument('--z-lim', type=float, nargs=2, metavar=('Z_MIN', 'Z_MAX'), default=[0, 10])
    parser.add_argument('--muram-tau-layer', type=float, default=0.1)
    parser.add_argument('--squashing-offset', type=float, default=1.5, help='Bottom crop in Mm for Q and twist.')
    parser.add_argument('--q-vmax', type=float, default=1e3)
    return parser.parse_args()


def _load_npz(path):
    npz_data = np.load(path, allow_pickle=True)
    return {key: npz_data[key] for key in npz_data.files}


def _get_mm_per_pixel(data, coords=None):
    if 'Mm_per_pixel' in data:
        return float(np.asarray(data['Mm_per_pixel']))
    if coords is not None and coords.shape[2] > 1:
        return float(np.nanmedian(np.diff(coords[0, 0, :, 2])))
    raise KeyError('Missing Mm_per_pixel and unable to infer it from coords.')


def _coords_extent(coords, z_count):
    coords = coords[:, :, :z_count]
    return (
        float(np.nanmin(coords[..., 0])),
        float(np.nanmax(coords[..., 0])),
        float(np.nanmin(coords[..., 1])),
        float(np.nanmax(coords[..., 1])),
        float(np.nanmin(coords[..., 2])),
        float(np.nanmax(coords[..., 2])),
    )


def _slice_index(x_slice, x_min, x_max, n_x):
    return int(np.argmin(np.abs(np.linspace(x_min, x_max, n_x) - x_slice)))


def _compute_integrated_current_map(data, z_count, mm_per_pixel):
    if 'j' not in data:
        raise KeyError(f'Missing j array in input data. Re-run nf2_to_npz with --metrics "j": {data.keys()}')
    m_per_pixel = mm_per_pixel * 1e6
    return np.linalg.norm(data['j'][:, :, :z_count], axis=-1).sum(axis=2) * m_per_pixel


def _sample_tau_bz(data, tau_target):
    if 'tau' not in data:
        raise KeyError('Missing tau array in MURaM data.')
    tau_index = np.argmin(np.abs(data['tau'] - tau_target), axis=2)
    x_idx, y_idx = np.meshgrid(np.arange(tau_index.shape[0]), np.arange(tau_index.shape[1]), indexing='ij')
    return data['b'][x_idx, y_idx, tau_index, 2]


def _dummy_squashing(data, z_count, offset):
    shape = (1, data['b'].shape[1], z_count - offset)
    return {'q': np.ones(shape), 'twist': np.zeros(shape)}


def _plot_integrated_current(data, ax, x_min, x_max, y_min, y_max, norm):
    im = ax.imshow(
        data.T,
        origin='lower',
        cmap='plasma',
        extent=[x_min, x_max, y_min, y_max],
        norm=norm,
    )
    return im


def _plot_bz_boundary(data, ax, x_min, x_max, y_min, y_max, vmax):
    im = ax.imshow(
        data.T,
        origin='lower',
        cmap='gray',
        vmin=-vmax,
        vmax=vmax,
        extent=[x_min, x_max, y_min, y_max],
    )
    return im


def _plot_b_nabla_bz(data, ax, x_slice_pix, y_min, y_max, z_min, z_max, z_count):
    im = ax.imshow(
        data[x_slice_pix, :, :z_count].T,
        origin='lower',
        cmap='bwr',
        vmin=-0.1,
        vmax=0.1,
        extent=[y_min, y_max, z_min, z_max],
    )
    return im


def _plot_squashing_factor_q(data, ax, y_min, y_max, z_min, z_max, z_offset, q_vmax):
    im = ax.imshow(
        data[0].T,
        origin='lower',
        cmap='viridis',
        extent=[y_min, y_max, z_min + z_offset, z_max],
        norm=LogNorm(vmin=1, vmax=q_vmax),
    )
    return im


def _plot_twist(data, ax, y_min, y_max, z_min, z_max, z_offset):
    im = ax.imshow(
        data[0].T,
        origin='lower',
        cmap='Spectral_r',
        extent=[y_min, y_max, z_min + z_offset, z_max],
        vmin=-1,
        vmax=1,
    )
    return im


def _add_group_label(fig, left_ax, right_ax, label, y_offset=0.055):
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    fig.text(
        (left_pos.x0 + right_pos.x1) / 2,
        left_pos.y1 + y_offset,
        label,
        ha='center',
        va='center',
        fontsize=13,
    )


def _output_path(output_file, suffix):
    root, ext = os.path.splitext(output_file)
    return f'{root}_{suffix}{ext or ".png"}'


def _set_xy_panel_labels(panel_axs):
    panel_axs[0, 0].set_ylabel('Y [Mm]')
    panel_axs[1, 0].set_ylabel('Y [Mm]')
    for ax in panel_axs[0]:
        ax.tick_params(axis='x', labelbottom=False)
    for ax in panel_axs[1]:
        ax.set_xlabel('X [Mm]')
    for ax in panel_axs[:, 1:].ravel():
        ax.tick_params(axis='y', labelleft=False, left=True)


def _add_yz_slice_markers(panel_axs, x_slice, y_lim):
    for ax in panel_axs.ravel():
        ax.plot(
            [x_slice, x_slice],
            y_lim,
            color='white',
            linestyle='--',
            linewidth=1.1,
            alpha=0.9,
        )


def _set_yz_panel_labels(panel_axs):
    for ax in panel_axs[:, 0]:
        ax.set_ylabel('Z [Mm]')
    for ax in panel_axs[-1]:
        ax.set_xlabel('Y [Mm]')
    for ax in panel_axs[:-1].ravel():
        ax.set_xticklabels([])
    for ax in panel_axs[:, 1:].ravel():
        ax.tick_params(axis='y', labelleft=False, left=True)


def _add_lambda_labels(fig, panel_axs, y_offset=0.055):
    labels = [
        r'$\lambda_\mathrm{ff}=1 \times 10^{-3}$',
        r'$\lambda_\mathrm{ff}=5 \times 10^{-4}$',
        r'$\lambda_\mathrm{ff}=1 \times 10^{-4}$',
        r'$\lambda_\mathrm{ff}=5 \times 10^{-5}$',
        r'$\lambda_\mathrm{ff}=1 \times 10^{-5}$',
    ]
    for pair_idx, label in enumerate(labels):
        left_col = 2 * pair_idx + 1
        _add_group_label(fig, panel_axs[0, left_col], panel_axs[0, left_col + 1], label, y_offset=y_offset)


def _create_panel_grid(n_rows, height):
    width_ratios = [1, 0.05, 1, 1, 0.05, 1, 1, 0.05, 1, 1, 0.05, 1, 1, 0.05, 1, 1, 0.06]
    panel_cols = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]
    fig = plt.figure(figsize=(25, height))
    gs = gridspec.GridSpec(n_rows, 17, width_ratios=width_ratios, wspace=0.13, hspace=0.12)
    fig.subplots_adjust(top=0.82 if n_rows == 2 else 0.95)
    axs = np.empty((n_rows, len(panel_cols)), dtype=object)
    for row in range(n_rows):
        for col_idx, col in enumerate(panel_cols):
            sharex = axs[0, col_idx] if n_rows == 2 and row == 1 else None
            axs[row, col_idx] = fig.add_subplot(gs[row, col], sharex=sharex)
    return fig, gs, axs


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    datasets = [
        Dataset('MURaM', _load_npz(args.data_muram)),
        Dataset(r'single-height', _load_npz(args.data_1slices_ff1e_3)),
        Dataset(r'multi-height', _load_npz(args.data_2slices_ff1e_3)),
        Dataset(r'single-height', _load_npz(args.data_1slices_ff5e_4)),
        Dataset(r'multi-height', _load_npz(args.data_2slices_ff5e_4)),
        Dataset(r'single-height', _load_npz(args.data_1slices_ff1e_4)),
        Dataset(r'multi-height', _load_npz(args.data_2slices_ff1e_4)),
        Dataset(r'single-height', _load_npz(args.data_1slices_ff5e_5)),
        Dataset(r'multi-height', _load_npz(args.data_2slices_ff5e_5)),
        Dataset(r'single-height', _load_npz(args.data_1slices_ff1e_5)),
        Dataset(r'multi-height', _load_npz(args.data_2slices_ff1e_5)),
    ]

    reference = datasets[1].data
    coords = reference['coords']
    z_count = min(
        coords.shape[2],
        *(dataset.data['b'].shape[2] for dataset in datasets),
        *(dataset.data['b_nabla_bz'].shape[2] for dataset in datasets),
    )
    x_min, x_max, y_min, y_max, z_min, z_max = _coords_extent(coords, z_count)
    x_slice_pix = _slice_index(args.x_slice, x_min, x_max, coords.shape[0])

    mm_per_pixel = _get_mm_per_pixel(datasets[0].data, coords=coords)
    offset = int(args.squashing_offset / mm_per_pixel)
    z_offset = offset * mm_per_pixel
    if offset >= z_count:
        raise ValueError(
            f'squashing-offset={args.squashing_offset} Mm removes the full cube height '
            f'({z_count} pixels at {mm_per_pixel} Mm/pixel).'
        )

    for dataset in datasets:
        dataset.bz_boundary_map = (
            _sample_tau_bz(dataset.data, args.muram_tau_layer)
            if dataset.title == 'MURaM'
            else dataset.data['b'][:, :, 0, 2]
        )
        dataset.current_map = _compute_integrated_current_map(dataset.data, z_count, mm_per_pixel)
        if USE_DUMMY_Q_T:
            dataset.squashing = _dummy_squashing(dataset.data, z_count, offset)
        else:
            from nf2.evaluation.output_metrics import squashing_factor

            dataset.squashing = squashing_factor(
                dataset.data['b'][:, :, offset:z_count],
                x_range=[x_slice_pix, x_slice_pix + 1],
            )

    current_values = np.concatenate([dataset.current_map.ravel() for dataset in datasets])
    current_values = current_values[np.isfinite(current_values) & (current_values > 0)]
    current_norm = LogNorm(vmin=current_values.min(), vmax=current_values.max())
    bz_vmax = max(float(np.nanmax(np.abs(dataset.bz_boundary_map))) for dataset in datasets)

    fig_xy, gs_xy, xy_axs = _create_panel_grid(2, 2.6)
    cax_bz = fig_xy.add_subplot(gs_xy[0, 16])
    cax_current = fig_xy.add_subplot(gs_xy[1, 16])

    for col_idx, dataset in enumerate(datasets):
        xy_axs[0, col_idx].set_title(dataset.title)
        im_bz = _plot_bz_boundary(dataset.bz_boundary_map, xy_axs[0, col_idx], x_min, x_max, y_min, y_max, bz_vmax)
        im_current = _plot_integrated_current(
            dataset.current_map, xy_axs[1, col_idx], x_min, x_max, y_min, y_max, current_norm
        )

    for ax in xy_axs.ravel():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    _add_yz_slice_markers(xy_axs, args.x_slice, args.y_lim)

    _set_xy_panel_labels(xy_axs)

    fig_xy.colorbar(im_bz, cax=cax_bz, location='right', label=r'$B_z$ [G]')
    fig_xy.colorbar(im_current, cax=cax_current, location='right', label=r'$\int |\mathbf{J}| dz$ [G$^2$ m / s]')

    _add_lambda_labels(fig_xy, xy_axs, y_offset=0.13)
    fig_xy.savefig(_output_path(args.output_file, 'xy'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_xy)

    fig_yz, gs_yz, yz_axs = _create_panel_grid(3, 6.4)
    cax_bnbz = fig_yz.add_subplot(gs_yz[0, 16])
    cax_q = fig_yz.add_subplot(gs_yz[1, 16])
    cax_twist = fig_yz.add_subplot(gs_yz[2, 16])

    for col_idx, dataset in enumerate(datasets):
        im_bnbz = _plot_b_nabla_bz(
            dataset.data['b_nabla_bz'],
            yz_axs[0, col_idx],
            x_slice_pix,
            y_min,
            y_max,
            z_min,
            z_max,
            z_count,
        )
        im_q = _plot_squashing_factor_q(
            dataset.squashing['q'], yz_axs[1, col_idx], y_min, y_max, z_min, z_max, z_offset, args.q_vmax
        )
        im_twist = _plot_twist(
            dataset.squashing['twist'], yz_axs[2, col_idx], y_min, y_max, z_min, z_max, z_offset
        )

    for ax in yz_axs.ravel():
        ax.set_xlim(args.y_lim)
        ax.set_ylim(args.z_lim)

    _set_yz_panel_labels(yz_axs)

    fig_yz.colorbar(im_bnbz, cax=cax_bnbz, location='right', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [1/Mm]')
    fig_yz.colorbar(im_q, cax=cax_q, location='right', label=r'$Q$')
    fig_yz.colorbar(im_twist, cax=cax_twist, location='right', label='Twist Number')

    fig_yz.savefig(_output_path(args.output_file, 'yz'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_yz)


if __name__ == '__main__':
    main()
