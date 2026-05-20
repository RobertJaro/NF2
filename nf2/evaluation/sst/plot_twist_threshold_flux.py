import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from nf2.evaluation.output_metrics import squashing_factor


DATASETS = [
    ('data_muram', 'MURaM'),
    ('data_1slice', '1 slice'),
    ('data_2slices', '2 slices'),
    ('data_2slices_ambiguous', '2 slices amb.'),
]
TWIST_CONTOURS = [0.5]
CM_PER_MM = 1e8


def _build_parser():
    parser = argparse.ArgumentParser(description='Compare magnetic-field and twist maps for MURaM and NF2 cubes.')
    parser.add_argument('--output', required=True, help='Output directory.')
    parser.add_argument('--data-muram', required=True, help='Path to the MURaM NPZ cube.')
    parser.add_argument('--data-1slice', required=True, help='Path to the 1-slice NF2 NPZ cube.')
    parser.add_argument('--data-2slices', required=True, help='Path to the 2-slices NF2 NPZ cube.')
    parser.add_argument('--data-2slices-ambiguous', required=True, help='Path to the ambiguous 2-slices NF2 NPZ cube.')
    parser.add_argument('--twist_height', type=float, default=1.5, help='Height in Mm where the twist map is evaluated.')
    parser.add_argument(
        '--truncate_height',
        type=float,
        default=None,
        help='Optional height in Mm below which the cube is removed before computing the twist map. Defaults to twist_height.',
    )
    parser.add_argument(
        '--field_slice',
        type=int,
        default=None,
        help='Optional slice index override for the top-row Bz maps. Defaults to the twist slice.',
    )
    parser.add_argument(
        '--sub_region',
        '--subdomain',
        type=float,
        nargs=4,
        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
        default=None,
        help='Optional centered-coordinate subdomain in Mm for the twist computation and plots: x_min x_max y_min y_max.',
    )
    return parser


def _load_npz(filepath):
    return dict(np.load(filepath))


def _load_axes(data):
    b_cube = data['b']

    if 'coords' in data:
        coords = data['coords']
        x_coords = coords[:, 0, 0, 0]
        y_coords = coords[0, :, 0, 1]
        z_coords = coords[0, 0, :, 2]
    else:
        Mm_per_pixel = float(data['Mm_per_pixel'])
        x_coords = (np.arange(b_cube.shape[0]) - (b_cube.shape[0] - 1) / 2) * Mm_per_pixel
        y_coords = (np.arange(b_cube.shape[1]) - (b_cube.shape[1] - 1) / 2) * Mm_per_pixel
        z_coords = np.arange(b_cube.shape[2]) * Mm_per_pixel

    return x_coords, y_coords, z_coords


def _axis_spacing(axis_coords):
    if axis_coords.size < 2:
        raise ValueError('At least two coordinate samples are required to determine the coordinate spacing')
    return float(np.nanmean(np.diff(axis_coords)))


def _select_slice(height_Mm, z_coords):
    n_slices = z_coords.shape[0]
    heights = z_coords
    slice_idx = int(np.argmin(np.abs(heights - height_Mm)))
    if not 0 <= slice_idx < n_slices:
        raise ValueError(f'slice={slice_idx} is outside the loaded cube range [0, {n_slices - 1}]')
    return slice_idx


def _validate_slice(slice_idx, n_slices, label):
    if not 0 <= slice_idx < n_slices:
        raise ValueError(f'{label}={slice_idx} is outside the loaded cube range [0, {n_slices - 1}]')
    return slice_idx


def _axis_extent(axis_coords, spacing=None):
    spacing = _axis_spacing(axis_coords) if spacing is None else spacing
    half_spacing = abs(spacing) / 2
    return [float(axis_coords[0] - half_spacing), float(axis_coords[-1] + half_spacing)]


def _select_axis_range(range_Mm, axis_coords, axis_label):
    axis_min_Mm, axis_max_Mm = range_Mm
    axis_low, axis_high = _axis_extent(axis_coords)

    if axis_min_Mm < axis_low or axis_max_Mm > axis_high:
        raise ValueError(
            f'{axis_label} sub_region [{axis_min_Mm}, {axis_max_Mm}] Mm is outside '
            f'the loaded cube range [{axis_low:.3f}, {axis_high:.3f}] Mm'
        )
    if axis_min_Mm >= axis_max_Mm:
        raise ValueError(f'{axis_label} sub_region minimum must be smaller than maximum')

    selected = np.where((axis_coords >= axis_min_Mm) & (axis_coords <= axis_max_Mm))[0]
    if selected.size == 0:
        raise ValueError(f'{axis_label} sub_region does not include any pixels')

    return int(selected[0]), int(selected[-1] + 1)


def _select_sub_region(sub_region, x_coords, y_coords):
    if sub_region is None:
        x_range = [0, x_coords.shape[0]]
        y_range = [0, y_coords.shape[0]]
    else:
        x_range = list(_select_axis_range(sub_region[:2], x_coords, 'x'))
        y_range = list(_select_axis_range(sub_region[2:], y_coords, 'y'))

    x_extent = _axis_extent(x_coords[x_range[0]:x_range[1]], _axis_spacing(x_coords))
    y_extent = _axis_extent(y_coords[y_range[0]:y_range[1]], _axis_spacing(y_coords))
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]
    return x_range, y_range, extent


def _prepare_dataset(name, data, twist_height, truncate_height, field_slice_override, sub_region, quantity_set='plot'):
    b_cube = data['b']
    x_coords, y_coords, z_coords = _load_axes(data)
    twist_slice = _select_slice(twist_height, z_coords)
    field_slice = twist_slice if field_slice_override is None else _validate_slice(field_slice_override, b_cube.shape[2], 'field_slice')
    truncate_height = twist_height if truncate_height is None else truncate_height
    truncate_slice = _select_slice(truncate_height, z_coords)
    if twist_slice < truncate_slice:
        raise ValueError(
            f'twist_height={twist_height} Mm is below truncate_height={truncate_height} Mm; '
            'the twist slice must be inside the height-cropped volume'
        )
    x_range, y_range, extent = _select_sub_region(sub_region, x_coords, y_coords)
    pixel_area_Mm2 = abs(_axis_spacing(x_coords) * _axis_spacing(y_coords))
    twist_cube = b_cube[x_range[0]:x_range[1], y_range[0]:y_range[1], truncate_slice:]
    relative_twist_slice = twist_slice - truncate_slice
    twist_map = squashing_factor(twist_cube, z_range=[relative_twist_slice, relative_twist_slice + 1])['twist'][:, :, 0]
    field_bz_map = b_cube[x_range[0]:x_range[1], y_range[0]:y_range[1], field_slice, 2]

    return {
        'name': name,
        'quantity_set': quantity_set,
        'field_bz_map': field_bz_map,
        'field_slice': field_slice,
        'field_height_Mm': z_coords[field_slice],
        'twist_map': twist_map,
        'twist_slice': twist_slice,
        'twist_height_Mm': z_coords[twist_slice],
        'truncate_slice': truncate_slice,
        'truncate_height_Mm': z_coords[truncate_slice],
        'sub_region_pixels': [x_range[0], x_range[1], y_range[0], y_range[1]],
        'pixel_area_Mm2': pixel_area_Mm2,
        'extent': extent,
    }


def _compute_threshold_flux(dataset, twist_threshold):
    mask = (
        np.isfinite(dataset['twist_map'])
        & np.isfinite(dataset['field_bz_map'])
        & (np.abs(dataset['twist_map']) >= twist_threshold)
    )
    area_Mm2 = dataset['pixel_area_Mm2'] * mask.sum()
    area_cm2 = area_Mm2 * CM_PER_MM ** 2
    pixel_area_cm2 = dataset['pixel_area_Mm2'] * CM_PER_MM ** 2
    signed_flux_Mx = np.nansum(dataset['field_bz_map'][mask]) * pixel_area_cm2
    unsigned_flux_Mx = np.nansum(np.abs(dataset['field_bz_map'][mask])) * pixel_area_cm2

    return {
        'mask_pixels': int(mask.sum()),
        'mask_area_Mm2': area_Mm2,
        'mask_area_cm2': area_cm2,
        'signed_flux_Mx': signed_flux_Mx,
        'unsigned_flux_Mx': unsigned_flux_Mx,
    }


def _write_flux_quantities(output_dir, datasets, twist_threshold):
    output_path = os.path.join(output_dir, 'twist_threshold_flux_quantities.txt')
    header = [
        'quantity_set',
        'dataset',
        'twist_threshold',
        'mask_pixels',
        'mask_area_Mm2',
        'mask_area_cm2',
        'signed_flux_Mx',
        'unsigned_flux_Mx',
        'field_height_Mm',
        'twist_height_Mm',
        'truncate_height_Mm',
        'pixel_area_Mm2',
    ]
    lines = ['\t'.join(header)]
    for dataset in datasets:
        quantities = _compute_threshold_flux(dataset, twist_threshold)
        row = [
            dataset['quantity_set'],
            dataset['name'],
            f'{twist_threshold:.6g}',
            str(quantities['mask_pixels']),
            f"{quantities['mask_area_Mm2']:.10e}",
            f"{quantities['mask_area_cm2']:.10e}",
            f"{quantities['signed_flux_Mx']:.10e}",
            f"{quantities['unsigned_flux_Mx']:.10e}",
            f"{dataset['field_height_Mm']:.10e}",
            f"{dataset['twist_height_Mm']:.10e}",
            f"{dataset['truncate_height_Mm']:.10e}",
            f"{dataset['pixel_area_Mm2']:.10e}",
        ]
        lines.append('\t'.join(row))

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


def _prepare_twist_height_zero_datasets(args):
    datasets = []
    for arg_name, label in DATASETS:
        if arg_name == 'data_muram':
            continue
        datasets.append(
            _prepare_dataset(
                label,
                _load_npz(getattr(args, arg_name)),
                twist_height=0,
                truncate_height=0,
                field_slice_override=args.field_slice,
                sub_region=args.sub_region,
                quantity_set='twist_height_0',
            )
        )
    return datasets


def _plot_comparison(output_dir, datasets):
    n_datasets = len(datasets)
    fig = plt.figure(figsize=(4, 1.7 * n_datasets), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        n_datasets + 1,
        2,
        height_ratios=[0.055] + [1] * n_datasets,
    )
    bz_cbar_ax = fig.add_subplot(grid_spec[0, 0])
    twist_cbar_ax = fig.add_subplot(grid_spec[0, 1])
    axs = np.empty((n_datasets, 2), dtype=object)
    shared_ax = None
    for row_idx in range(n_datasets):
        for col_idx in range(2):
            axs[row_idx, col_idx] = fig.add_subplot(
                grid_spec[row_idx + 1, col_idx],
                sharex=shared_ax,
                sharey=shared_ax,
            )
            if shared_ax is None:
                shared_ax = axs[row_idx, col_idx]

    bz_limit = max(np.nanmax(np.abs(dataset['field_bz_map'])) for dataset in datasets)
    bz_norm = Normalize(vmin=-bz_limit, vmax=bz_limit)
    twist_norm = Normalize(vmin=-2, vmax=2)

    bz_image = None
    twist_image = None

    for row_idx, dataset in enumerate(datasets):
        ax = axs[row_idx, 0]
        abs_twist_map = np.abs(dataset['twist_map'])
        bz_image = ax.imshow(
            dataset['field_bz_map'].T,
            origin='lower',
            extent=dataset['extent'],
            cmap='gray',
            norm=bz_norm,
        )
        ax.contour(
            abs_twist_map.T,
            levels=TWIST_CONTOURS,
            colors=['tab:orange', 'tab:red'],
            linewidths=1.0,
            origin='lower',
            extent=dataset['extent'],
        )

        ax = axs[row_idx, 1]
        twist_image = ax.imshow(
            dataset['twist_map'].T,
            origin='lower',
            extent=dataset['extent'],
            cmap='coolwarm',
            norm=twist_norm,
        )

    for ax in axs.flat:
        ax.label_outer()

    for ax in axs[:, 0]:
        ax.set_ylabel('Y [Mm]')
    for ax in axs[-1, :]:
        ax.set_xlabel('X [Mm]')

    bz_cbar = fig.colorbar(bz_image, cax=bz_cbar_ax, orientation='horizontal')
    bz_cbar.set_label('Bz [G]')
    bz_cbar_ax.xaxis.set_ticks_position('top')
    bz_cbar_ax.xaxis.set_label_position('top')

    twist_cbar = fig.colorbar(twist_image, cax=twist_cbar_ax, orientation='horizontal')
    twist_cbar.set_label('Twist')
    twist_cbar_ax.xaxis.set_ticks_position('top')
    twist_cbar_ax.xaxis.set_label_position('top')

    plot_path = os.path.join(output_dir, 'twist_maps.png')
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    datasets = [
        _prepare_dataset(
            label,
            _load_npz(getattr(args, arg_name)),
            args.twist_height,
            args.truncate_height,
            args.field_slice,
            args.sub_region,
        )
        for arg_name, label in DATASETS
    ]

    _plot_comparison(args.output, datasets)
    flux_datasets = datasets + _prepare_twist_height_zero_datasets(args)
    _write_flux_quantities(args.output, flux_datasets, TWIST_CONTOURS[0])


if __name__ == '__main__':
    main()
