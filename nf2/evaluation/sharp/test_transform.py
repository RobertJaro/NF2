import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

from nf2.train.model import image_to_spherical_matrix
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

if __name__ == '__main__':
    B_field_map = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.field.fits')
    B_az_map = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.azimuth.fits')
    B_disamb_map = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.disambig.fits')
    B_in_map = Map('/glade/work/rjarolim/data/SST/sharp/hmi.b_720s.20180930_092400_TAI.inclination.fits')

    fld = B_field_map.data
    inc = np.deg2rad(B_in_map.data)
    azi = np.deg2rad(B_az_map.data)
    amb = B_disamb_map.data
    # disambiguate
    amb_weak = 2
    condition = (amb.astype((int)) >> amb_weak).astype(bool)
    azi[condition] += np.pi

    sin = np.sin
    cos = np.cos
    B_x = - fld * sin(inc) * sin(azi)
    B_y = fld * sin(inc) * cos(azi)
    B_z = fld * cos(inc)

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs[0, 0].imshow(B_x, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    axs[0, 0].set_title('Bx')
    axs[0, 1].imshow(B_field_map.data, cmap='gray', origin='lower', vmin=0, vmax=1000)
    axs[0, 1].set_title('B')
    #
    axs[1, 0].imshow(B_y, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    axs[1, 0].set_title('By')
    axs[1, 1].imshow(B_in_map.data, cmap='gray', origin='lower', vmin=0, vmax=180)
    axs[1, 1].set_title('Inclination')
    #
    axs[2, 0].imshow(B_z, cmap='gray', origin='lower', vmin=-1000, vmax=1000)
    axs[2, 0].set_title('Bz')
    axs[2, 1].imshow(B_az_map.data, cmap='gray', origin='lower', vmin=0, vmax=360)
    axs[2, 1].set_title('Azimuth')
    plt.tight_layout()
    plt.savefig('/glade/work/rjarolim/data/SST/sharp/image_frame.jpg', dpi=150)
    plt.close()

    B = np.stack([B_x, B_y, B_z], -1)
    latc, lonc = np.deg2rad(B_field_map.meta['CRLT_OBS']), np.deg2rad(B_field_map.meta['CRLN_OBS'])

    coords = all_coordinates_from_map(B_field_map).transform_to(frames.HeliographicCarrington)
    lat, lon = coords.lat.to(u.rad).value, coords.lon.to(u.rad).value

    pAng = -np.deg2rad(B_field_map.meta['CROTA2'])

    a_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
    Brtp = np.einsum('...ij,...j->...i', a_matrix, B)
    Brtp = np.flip(Brtp, axis=(0,1)) # match SHARPs orientation
    bp = Brtp[..., 2]
    bt = -Brtp[..., 1]
    br = Brtp[..., 0]

    # to grid coords

    ref_B_p = Map('/glade/work/rjarolim/data/SST/sharp/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Bp.fits').data
    ref_B_t = Map('/glade/work/rjarolim/data/SST/sharp/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Bt.fits').data
    ref_B_r = Map('/glade/work/rjarolim/data/SST/sharp/hmi.sharp_cea_720s.7310.20180930_092400_TAI.Br.fits').data

    # plot
    norm = SymLogNorm(linthresh=1, vmin=-2000, vmax=2000)
    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs[0, 0].imshow(bp, cmap='RdBu_r', origin='lower', norm=norm)
    axs[0, 0].set_title('Bp')
    axs[0, 1].imshow(ref_B_p, cmap='RdBu_r', origin='lower', norm=norm)
    axs[0, 1].set_title('ref Bp')
    axs[1, 0].imshow(bt, cmap='RdBu_r', origin='lower', norm=norm)
    axs[1, 0].set_title('Bt')
    axs[1, 1].imshow(ref_B_t, cmap='RdBu_r', origin='lower', norm=norm)
    axs[1, 1].set_title('ref Bt')
    axs[2, 0].imshow(br, cmap='RdBu_r', origin='lower', norm=norm)
    axs[2, 0].set_title('Br')
    axs[2, 1].imshow(ref_B_r, cmap='RdBu_r', origin='lower', norm=norm)
    axs[2, 1].set_title('ref Br')
    plt.tight_layout()
    fig.savefig('/glade/work/rjarolim/data/SST/sharp/rtp_frame.jpg', dpi=300)
    plt.close(fig)

    raise Exception('stop here')

    # transform back to az and inc
    ai_matrix = np.linalg.inv(a_matrix)
    B = np.einsum('...ij,...j->...i', ai_matrix, Brtp)
    # B_x = - fld * sin(inc) * sin(azi)
    # B_y = fld * sin(inc) * cos(azi)
    # B_z = fld * cos(inc)
    eps = 1e-7
    fld_rec = np.sqrt(B[..., 0] ** 2 + B[..., 1] ** 2 + B[..., 2] ** 2)
    inc_rec = np.arccos(np.clip(B[..., 2] / (fld + eps), a_min=-1 + eps, a_max=1 - eps))
    azi_rec = np.arctan2(B[..., 0], -B[..., 1] + eps)

    # plot
    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs[0, 0].imshow(fld_rec, cmap='gray', origin='lower', vmin=0, vmax=1000)
    axs[0, 0].set_title('B')
    axs[0, 1].imshow(fld, cmap='gray', origin='lower', vmin=0, vmax=1000)
    axs[0, 1].set_title('ref B')
    axs[1, 0].imshow(inc_rec, cmap='gray', origin='lower', vmin=0, vmax=np.pi / 2)
    axs[1, 0].set_title('inc')
    axs[1, 1].imshow(inc, cmap='gray', origin='lower', vmin=0, vmax=np.pi / 2)
    axs[1, 1].set_title('ref inc')
    axs[2, 0].imshow(azi_rec, cmap='gray', origin='lower', vmin=-np.pi, vmax=np.pi)
    axs[2, 0].set_title('azi')
    axs[2, 1].imshow(azi - np.pi, cmap='gray', origin='lower', vmin=-np.pi, vmax=np.pi)
    axs[2, 1].set_title('ref azi')
    plt.tight_layout()
    plt.savefig('/gpfs/gpfs0/robert.jarolim/data/nf2/fd_2154/full_disk/azinc_frame.jpg', dpi=150)
    plt.close()

    # plot
    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs[0, 0].imshow(fld_rec, cmap='gray', origin='lower', vmin=0, vmax=1000)
    axs[0, 0].set_title('B')
    axs[0, 1].imshow(fld, cmap='gray', origin='lower', vmin=0, vmax=1000)
    axs[0, 1].set_title('ref B')
    axs[1, 0].imshow(inc_rec, cmap='gray', origin='lower', vmin=0, vmax=np.pi / 2)
    axs[1, 0].set_title('inc')
    axs[1, 1].imshow(inc, cmap='gray', origin='lower', vmin=0, vmax=np.pi / 2)
    axs[1, 1].set_title('ref inc')
    axs[2, 0].imshow(np.cos(2 * azi_rec), cmap='gray', origin='lower', vmin=-1, vmax=1)
    axs[2, 0].set_title('azi')
    axs[2, 1].imshow(np.cos(2 * np.deg2rad(B_az_map.data)), cmap='gray', origin='lower', vmin=-1, vmax=1)
    axs[2, 1].set_title('ref azi')
    plt.tight_layout()
    fig.savefig('/gpfs/gpfs0/robert.jarolim/data/nf2/fd_2154/full_disk/disamb.jpg', dpi=150)
    plt.close()
