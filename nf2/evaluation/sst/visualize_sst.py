from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.visualization.colormaps import cm


MAGNETOGRAM_LIMIT = 1000
INTENSITY_VMIN = 0.1e-8
INTENSITY_VMAX = 3e-8
MM_PER_PIXEL_INTENSITY = 0.045
OUTPUT_DIR = Path('/glade/work/rjarolim/nf2/topology/results/observations')
LABEL_SIZE = 11
TICK_SIZE = 10

CONFIGS = [
    {
        'photosphere_blos_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_v2/BLOS_panorama0851_6173_StkIQUV_rebin.fits'),
        'chromosphere_blos_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_v2/BLOS_panorama0851_8542_StkIQUV_rebin.fits'),
        'chromosphere_mask_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_v2/mask.fits'),
        'intensity_path': Path('/glade/work/rjarolim/data/SST/panorama_8542_StkI.fits'),
        'intensity_origin': 'lower',
        'output_name': '08_51.png',
        'slice_line_x': -60,
    },
    {
        'photosphere_blos_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_1050_converted/BLOS_panorama1050_6173_StkIQUV_rebin.fits'),
        'chromosphere_blos_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_1050_converted/BLOS_panorama1050_8542_StkIQUV_rebin.fits'),
        'chromosphere_mask_path': Path('/glade/work/rjarolim/data/SST/campaign_2023_1050_converted/mask.fits'),
        'intensity_path': Path('/glade/work/rjarolim/data/SST/panorama1050_8542_StkI.fits'),
        'intensity_origin': 'lower',
        'output_name': '10_50.png',
        'slice_line_x': None,
    },
]


def _extent(data, mm_per_pixel: float) -> list[float]:
    return [
        -data.shape[1] / 2 * mm_per_pixel,
        data.shape[1] / 2 * mm_per_pixel,
        -data.shape[0] / 2 * mm_per_pixel,
        data.shape[0] / 2 * mm_per_pixel,
    ]


def _add_top_colorbar(fig, ax, im, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4.5%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(label, size=LABEL_SIZE, labelpad=6)
    cbar.ax.tick_params(labelsize=TICK_SIZE, pad=1)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')


def _plot_dataset(dataset: dict) -> None:
    photosphere_blos = fits.getdata(dataset['photosphere_blos_path'])
    chromosphere_blos = fits.getdata(dataset['chromosphere_blos_path']).astype(float)
    chromosphere_mask = fits.getdata(dataset['chromosphere_mask_path'])
    intensity = fits.getdata(dataset['intensity_path'])[10]
    chromosphere_blos[chromosphere_mask != 0] = float('nan')

    photosphere_extent = _extent(photosphere_blos, 0.09)
    chromosphere_extent = _extent(chromosphere_blos, 0.09)
    intensity_extent = _extent(intensity, MM_PER_PIXEL_INTENSITY)

    fig, axs = plt.subplots(figsize=(8.6, 2.9), ncols=3)

    im = axs[0].imshow(
        photosphere_blos,
        cmap='gray',
        origin='lower',
        extent=photosphere_extent,
        vmin=-MAGNETOGRAM_LIMIT,
        vmax=MAGNETOGRAM_LIMIT,
    )
    axs[0].set_xlabel('X [Mm]', fontsize=LABEL_SIZE)
    axs[0].set_ylabel('Y [Mm]', fontsize=LABEL_SIZE)
    axs[0].tick_params(labelsize=TICK_SIZE)
    _add_top_colorbar(fig, axs[0], im, r'$B_\text{LOS}$ [G]')

    im = axs[1].imshow(
        chromosphere_blos,
        cmap='gray',
        origin='lower',
        extent=chromosphere_extent,
        vmin=-MAGNETOGRAM_LIMIT,
        vmax=MAGNETOGRAM_LIMIT,
    )
    axs[1].set_xlabel('X [Mm]', fontsize=LABEL_SIZE)
    axs[1].set_ylabel('')
    axs[1].set_yticklabels([])
    axs[1].tick_params(labelsize=TICK_SIZE)
    _add_top_colorbar(fig, axs[1], im, r'$B_\text{LOS}$ [G]')

    im = axs[2].imshow(
        intensity,
        cmap=cm.yohkohsxtal,
        origin=dataset['intensity_origin'],
        extent=intensity_extent,
        vmin=INTENSITY_VMIN,
        vmax=INTENSITY_VMAX,
    )
    axs[2].contour(
        photosphere_blos,
        levels=[-MAGNETOGRAM_LIMIT, MAGNETOGRAM_LIMIT],
        colors=['blue', 'red'],
        extent=photosphere_extent,
        alpha=0.6,
        linewidths=0.8,
    )
    axs[2].set_xlabel('X [Mm]', fontsize=LABEL_SIZE)
    axs[2].set_ylabel('')
    axs[2].set_yticklabels([])
    axs[2].tick_params(labelsize=TICK_SIZE)
    if dataset['slice_line_x'] is not None:
        axs[2].plot(
            [dataset['slice_line_x'], dataset['slice_line_x']],
            [-15, 15],
            color='orange',
            linestyle='--',
            linewidth=1,
        )
    _add_top_colorbar(fig, axs[2], im, 'Intensity [DN]')

    fig.tight_layout(pad=0.4, w_pad=0.5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / dataset['output_name'], dpi=300, transparent=True)
    plt.close(fig)


for dataset_config in CONFIGS:
    _plot_dataset(dataset_config)
