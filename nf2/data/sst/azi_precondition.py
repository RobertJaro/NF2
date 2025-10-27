import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from tqdm import tqdm

from nf2.potential.potential_field import get_potential_field, compute_potential

azi_files = sorted(glob.glob('/glade/work/rjarolim/data/nf2/nlte/BAZI_*.fits'))
los_files = sorted(glob.glob('/glade/work/rjarolim/data/nf2/nlte/BLOS_*.fits'))
trv_files = sorted(glob.glob('/glade/work/rjarolim/data/nf2/nlte/BTRV_*.fits'))


def wrap_minus_pi_to_pi(x):
    """Wrap angle(s) to (-pi, pi]. Works with scalars or numpy arrays."""
    return (x + np.pi) % (2*np.pi) - np.pi

def minimize_angular_gap(a, b):
    """
    Given angles a, b in [0, 2π), optionally flip b by π (mod 2π)
    if it reduces the circular distance to a. Returns (b_best, dist, flipped).
    """
    # distance without flip
    d0 = np.abs(wrap_minus_pi_to_pi(a - b))
    # distance if we flip b by π
    b_flip = (b + np.pi) % (2*np.pi)
    d1 = np.abs(wrap_minus_pi_to_pi(a - b_flip))

    flipped = d1 < d0
    # choose the better option
    b_best = np.where(flipped, b_flip, b)
    dist = np.where(flipped, d1, d0)
    return b_best, dist, flipped

for f_azi, f_los, f_trv in tqdm(zip(azi_files, los_files, trv_files), total=len(azi_files)):
    original_azi = (np.pi - fits.getdata(f_azi)).T
    data_los = fits.getdata(f_los).T
    data_trv = fits.getdata(f_trv).T

    b_n = data_los

    cube_shape = b_n.shape

    b_n = b_n.reshape((-1)).astype(np.float32)
    coords = [np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], -1:2], -1)]
    fields = compute_potential(coords, cube_shape, b_n)

    B_potential = fields[0][:, :, 1]

    azi_potential = np.arctan2(B_potential[..., 0], B_potential[..., 1]) % (2 * np.pi)
    azi_diff = np.abs(azi_potential - original_azi)
    preconditioned_azi = np.copy(original_azi)
    preconditioned_azi = minimize_angular_gap(azi_potential, preconditioned_azi)[0]
    preconditioned_azi = preconditioned_azi % (2 * np.pi)

    bx = np.sin(preconditioned_azi) * data_trv
    by = np.cos(preconditioned_azi) * data_trv
    bz = data_los

    bx_potential = B_potential[..., 0]
    by_potential = B_potential[..., 1]
    bz_potential = B_potential[..., 2]

    fig, axs = plt.subplots(3, 4, figsize=(15, 5))
    ax = axs[0, 0]
    im = ax.imshow(original_azi.T, vmin=0, vmax=np.pi, cmap='twilight', origin='lower')
    ax.set_title('Original Azimuth')
    plt.colorbar(im, ax=ax)

    ax = axs[0, 1]
    im = ax.imshow(azi_potential.T % (2 * np.pi), vmin=0, vmax=2 * np.pi, cmap='twilight', origin='lower')
    ax.set_title('Potential Field Azimuth')
    plt.colorbar(im, ax=ax)

    ax = axs[0, 2]
    im = ax.imshow(preconditioned_azi.T % (2 * np.pi), vmin=0, vmax=2 * np.pi, cmap='twilight', origin='lower')
    ax.set_title('Preconditioned Azimuth')
    plt.colorbar(im, ax=ax)

    ax = axs[0, 3]
    im = ax.imshow(preconditioned_azi.T % np.pi, vmin=0, vmax=np.pi, cmap='twilight', origin='lower')
    ax.set_title('Check')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 0]
    im = ax.imshow(bx.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_x')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 1]
    im = ax.imshow(by.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_y')
    plt.colorbar(im, ax=ax)

    ax = axs[1, 2]
    im = ax.imshow(bz.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_z')
    plt.colorbar(im, ax=ax)

    ax = axs[2, 0]
    im = ax.imshow(bx_potential.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_x Potential')
    plt.colorbar(im, ax=ax)

    ax = axs[2, 1]
    im = ax.imshow(by_potential.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_y Potential')
    plt.colorbar(im, ax=ax)

    ax = axs[2, 2]
    im = ax.imshow(bz_potential.T, vmin=-500, vmax=500, cmap='gray', origin='lower')
    ax.set_title('B_z Potential')
    plt.colorbar(im, ax=ax)


    f_img = f_azi.replace('BAZI', 'BAZIPRE').replace('.fits', '.jpg')
    fig.savefig(f_img)
    plt.close(fig)

    # write to fits
    header = fits.getheader(f_azi)
    new_file = f_azi.replace('BAZI', 'BAZIPRE')
    fits.writeto(new_file, np.pi - preconditioned_azi.T, header, overwrite=True)
