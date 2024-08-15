import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.util import vector_cartesian_to_spherical
from nf2.evaluation.metric import theta_J, divergence_jacobian, energy, b_diff_error
from nf2.evaluation.output import CartesianOutput, SphericalOutput
from nf2.evaluation.output_metrics import current_density, spherical_energy_gradient, energy_gradient
from nf2.potential.potential_field import get_potential_field

Mm_per_pixel = 0.72

cartesian_model = CartesianOutput('/glade/work/rjarolim/nf2/sharp/377_v01/extrapolation_result.nf2')
cartesian_out = cartesian_model.load_cube(progress=True, Mm_per_pixel=Mm_per_pixel, metrics={'j' : current_density, 'energy_gradient': energy_gradient})

reference_map = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits')

bottom_left = reference_map.bottom_left_coord.transform_to(frames.HeliographicCarrington)
top_right = reference_map.top_right_coord.transform_to(frames.HeliographicCarrington)
center = reference_map.center.transform_to(frames.HeliographicCarrington)

spherical_model = SphericalOutput('/glade/work/rjarolim/nf2/spherical/377_v05/extrapolation_result.nf2')
spherical_out = spherical_model.load_spherical(
    latitude_range=((np.pi / 2 * u.rad) - top_right.lat, (np.pi / 2 * u.rad) - bottom_left.lat),
    longitude_range=(bottom_left.lon, top_right.lon),
    radius_range=(1. * u.solRad, 1 * u.solRad + 100 * u.Mm),
    sampling=[cartesian_out['coords'].shape[2], cartesian_out['coords'].shape[1], cartesian_out['coords'].shape[0]],
    metrics={'j' : current_density, 'energy_gradient': spherical_energy_gradient},
    progress=True)

subframe_model = SphericalOutput('/glade/work/rjarolim/nf2/spherical/377_subframe_v04/extrapolation_result.nf2')
subframe_out = subframe_model.load_spherical(
    latitude_range=((np.pi / 2 * u.rad) - top_right.lat, (np.pi / 2 * u.rad) - bottom_left.lat),
    longitude_range=(bottom_left.lon, top_right.lon),
    radius_range=(1. * u.solRad, 1 * u.solRad + 100 * u.Mm),
    sampling=[cartesian_out['coords'].shape[2], cartesian_out['coords'].shape[1], cartesian_out['coords'].shape[0]],
    metrics={'j' : current_density, 'energy_gradient': spherical_energy_gradient},
    progress=True)

pb_model = SphericalOutput('/glade/work/rjarolim/nf2/spherical/377_potential_v01/extrapolation_result.nf2')
pb_out = spherical_model.load_spherical(
    latitude_range=((np.pi / 2 * u.rad) - top_right.lat, (np.pi / 2 * u.rad) - bottom_left.lat),
    longitude_range=(bottom_left.lon, top_right.lon),
    radius_range=(1. * u.solRad, 1 * u.solRad + 100 * u.Mm),
    sampling=[cartesian_out['coords'].shape[2], cartesian_out['coords'].shape[1], cartesian_out['coords'].shape[0]],
    metrics={'j' : current_density, 'energy_gradient': spherical_energy_gradient},
    progress=True)

b_pb = pb_out['b'].value
b_spherical = spherical_out['b'].value
b_cartesian = cartesian_out['b'].value
b_subframe = subframe_out['b'].value

j_pb = pb_out['j'].value
j_spherical = spherical_out['j'].value
j_cartesian = cartesian_out['j'].value
j_subframe = subframe_out['j'].value

energy_gradient_pb = pb_out['energy_gradient'].value
energy_gradient_spherical = spherical_out['energy_gradient'].value
energy_gradient_cartesian = cartesian_out['energy_gradient'].value
energy_gradient_subframe = subframe_out['energy_gradient'].value

coords_pb = pb_out['coords']
coords_spherical = spherical_out['coords']
coords_cartesian = cartesian_out['coords']
coords_subframe = subframe_out['coords']

div_pb = np.nanmean(np.abs(divergence_jacobian(pb_out['jac_matrix'].value)))
div_spherical = np.nanmean(np.abs(divergence_jacobian(spherical_out['jac_matrix'].value)))
div_cartesian = np.nanmean(np.abs(divergence_jacobian(cartesian_out['jac_matrix'].value)))
div_subframe = np.nanmean(np.abs(divergence_jacobian(subframe_out['jac_matrix'].value)))

theta_pb = theta_J(b_pb, j_pb)
theta_spherical = theta_J(b_spherical, j_spherical)
theta_cartesian = theta_J(b_cartesian, j_cartesian)
theta_subframe = theta_J(b_subframe, j_subframe)

bx_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits')
by_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits')
bz_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits')
b_cartesian_ref = np.stack([bx_ref.data, -by_ref.data, bz_ref.data]).T

errx_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp_err.fits')
erry_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt_err.fits')
errz_ref = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br_err.fits')
err_cartesian_ref = np.stack([errx_ref.data, erry_ref.data, errz_ref.data]).T

spherical_coords = all_coordinates_from_map(reference_map).T
#
img_out = cartesian_model.load_boundary()
b_cartesian_img = img_out['b'].value
#
img_out = spherical_model.load_spherical_coords(spherical_coords)
b_spherical_img = vector_cartesian_to_spherical(img_out['b'].value, img_out['spherical_coords'])
b_spherical_img = np.flip(b_spherical_img, -1)
b_spherical_img[..., 1] *= -1
#
img_out = subframe_model.load_spherical_coords(spherical_coords)
b_subframe_img = vector_cartesian_to_spherical(img_out['b'].value, img_out['spherical_coords'])
b_subframe_img = np.flip(b_subframe_img, -1)
b_subframe_img[..., 1] *= -1
#
img_out = pb_model.load_spherical_coords(spherical_coords)
b_pb_img = vector_cartesian_to_spherical(img_out['b'].value, img_out['spherical_coords'])
b_pb_img = np.flip(b_pb_img, -1)
b_pb_img[..., 1] *= -1

b_diff_cartesian = np.linalg.norm(b_cartesian_img - b_cartesian_ref, axis=-1).mean()
b_diff_spherical = np.linalg.norm(b_spherical_img - b_cartesian_ref, axis=-1).mean()
b_diff_subframe = np.linalg.norm(b_subframe_img - b_cartesian_ref, axis=-1).mean()
b_dff_pb = np.linalg.norm(b_pb_img - b_cartesian_ref, axis=-1).mean()

b_diff_err_cartesian = b_diff_error(b_cartesian_img, b_cartesian_ref, err_cartesian_ref).mean()
b_diff_err_spherical = b_diff_error(b_spherical_img, b_cartesian_ref, err_cartesian_ref).mean()
b_diff_err_subframe = b_diff_error(b_subframe_img, b_cartesian_ref, err_cartesian_ref).mean()
b_diff_err_pb = b_diff_error(b_pb_img, b_cartesian_ref, err_cartesian_ref).mean()

print(f'Divergence: Cartesian: {div_cartesian:.2e}, Spherical:  {div_spherical:.2e}, Subframe: {div_subframe:.2e}, PB: {div_pb:.2e}')
print(f'Weighted Theta:  Cartesian: {theta_cartesian:.2f}, Spherical: {theta_spherical:.2f}, Subframe: {theta_subframe:.2f}, PB: {theta_pb:.2f}')
print(f'B Diff: Cartesian: {b_diff_cartesian:.2f}, Spherical: {b_diff_spherical:.2f}, Subframe: {b_diff_subframe:.2f}, PB: {b_dff_pb:.2f}')
print(f'B Diff Err: Cartesian: {b_diff_err_cartesian:.2f}, Spherical: {b_diff_err_spherical:.2f}, Subframe: {b_diff_err_subframe:.2f}, PB: {b_diff_err_pb:.2f}')

# print(f'Saving Cube (spherical): {coords_spherical.shape}')
# save_vtk('/glade/work/rjarolim/nf2/spherical/evaluation/spherical.vtk', coords_spherical, vectors={'b': np.nan_to_num(b_spherical), 'j': np.nan_to_num(j_spherical)}, Mm_per_pix=0.72)
# print(f'Saving Cube (cartesian): {coords_cartesian.shape}')
# save_vtk('/glade/work/rjarolim/nf2/spherical/evaluation/cartesian.vtk', coords_cartesian, vectors={'b': b_cartesian, 'j': j_cartesian}, Mm_per_pix=0.72)


b_potential = get_potential_field(b_cartesian[:, :, 0, 2], b_cartesian.shape[2], batch_size=int(1e3))
cartesian_free_energy = energy(b_cartesian) - energy(b_potential)
sphere_free_energy = energy(b_spherical) - np.moveaxis(energy(b_potential), (0, 1, 2), (2, 1, 0))
subframe_free_energy = energy(b_subframe) - np.moveaxis(energy(b_potential), (0, 1, 2), (2, 1, 0))

# plot comparison of currents and jxb

cartesian_extent = [0, b_cartesian.shape[0] * Mm_per_pixel, 0, b_cartesian.shape[1] * Mm_per_pixel]
spherical_extent = [bottom_left.lon.to(u.deg).value, top_right.lon.to(u.deg).value,
          top_right.lat.to(u.deg).value, bottom_left.lat.to(u.deg).value, ]

fig, axs = plt.subplots(3, 3, figsize=(12, 8), )

ax = axs[0, 0]
im = ax.imshow(b_cartesian_img[..., 2].T, origin='lower', cmap='gray', extent=cartesian_extent, vmin=-1000, vmax=1000)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')
ax.set_title('Cartesian')
ax.set_ylabel('Y [Mm]')
ax.set_xlabel('X [Mm]')

ax = axs[0, 1]
im = ax.imshow(b_spherical_img[..., 2].T, origin='lower', cmap='gray', extent=spherical_extent, vmin=-1000, vmax=1000)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')
ax.set_title('Spherical')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')

ax = axs[0, 2]
im = ax.imshow(b_subframe_img[..., 2].T, origin='lower', cmap='gray', extent=spherical_extent, vmin=-1000, vmax=1000)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')
ax.set_title('Subframe')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')

norm = LogNorm()
ax = axs[1, 0]
im = ax.imshow(np.linalg.norm(j_cartesian, axis=-1).sum(2).T * Mm_per_pixel * 1e6, origin='lower',
                 cmap='inferno', norm=norm, extent=cartesian_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'|$\nabla x $B| [G])')
ax.set_title('Cartesian')
ax.set_ylabel('Y [Mm]')
ax.set_xlabel('X [Mm]')

ax = axs[1, 1]
im = ax.imshow(np.nansum(np.linalg.norm(j_spherical, axis=-1), 0) * Mm_per_pixel * 1e6, origin='upper',
                 cmap='inferno', norm=norm, extent=spherical_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'|$\nabla x $B| [G])')
ax.set_title('Spherical')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')

ax = axs[1, 2]
im = ax.imshow(np.nansum(np.linalg.norm(j_subframe, axis=-1), 0) * Mm_per_pixel * 1e6, origin='upper',
                    cmap='inferno', norm=norm, extent=spherical_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'|$\nabla x $B| [G])')
ax.set_title('Subframe')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')


norm = LogNorm(vmin=1e9)
ax = axs[2, 0]
im = ax.imshow(cartesian_free_energy.sum(2).T * Mm_per_pixel * 1e6, origin='lower', cmap='viridis', norm=norm,
                    extent=cartesian_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Energy [erg/cm$^3$]')
ax.set_title('Cartesian')
ax.set_ylabel('Y [Mm]')
ax.set_xlabel('X [Mm]')

ax = axs[2, 1]
im = ax.imshow(sphere_free_energy.sum(0) * Mm_per_pixel * 1e6, origin='upper', cmap='viridis', norm=norm,
                    extent=spherical_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Energy [erg/cm$^3$]')
ax.set_title('Spherical')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')

ax = axs[2, 2]
im = ax.imshow(subframe_free_energy.sum(0) * Mm_per_pixel * 1e6, origin='upper', cmap='viridis', norm=norm,
                    extent=spherical_extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Energy [erg/cm$^3$]')
ax.set_title('Subframe')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Longitude [deg]')

fig.tight_layout()
plt.savefig('/glade/work/rjarolim/nf2/spherical/evaluation/overview.png', dpi=300, transparent=True)
plt.close()




fig, axs = plt.subplots(1, 5, figsize=(15, 3), )

ax = axs[0]
cartesian_energy = energy(b_cartesian).sum((0, 1)) * Mm_per_pixel ** 2
spherical_energy = energy(b_spherical).sum((1, 2)) * Mm_per_pixel ** 2
subframe_energy = energy(b_subframe).sum((1, 2)) * Mm_per_pixel ** 2
pb_energy = energy(b_pb).sum((1, 2)) * Mm_per_pixel ** 2
height = np.linspace(0, b_cartesian.shape[2] * Mm_per_pixel, b_cartesian.shape[2])
ax.plot(cartesian_energy, height, label='Cartesian')
ax.plot(spherical_energy, height, label='Spherical')
ax.plot(subframe_energy, height, label='Subframe')
ax.plot(pb_energy, height, label='Potential Boundary')
ax.set_ylabel('Height [Mm]')
ax.set_xlabel('Energy [erg/Mm]')
ax.set_ylim(0, 100)
ax.semilogx()

# currents
ax = axs[1]
ax.plot(np.linalg.norm(j_cartesian, axis=-1).mean((0, 1)), height, label='Cartesian')
ax.plot(np.linalg.norm(j_spherical, axis=-1).mean((1, 2)), height, label='Spherical')
ax.plot(np.linalg.norm(j_subframe, axis=-1).mean((1, 2)), height, label='Subframe')
ax.plot(np.linalg.norm(j_pb, axis=-1).mean((1, 2)), height, label='Potential Boundary')
ax.set_ylabel('Height [Mm]')
ax.set_xlabel('|J| [G/Mm]')
ax.set_ylim(0, 100)
ax.semilogx()

ax = axs[2]
cartesian_free_energy_h = cartesian_free_energy.sum((0, 1)) * Mm_per_pixel ** 2
spherical_free_energy_h = sphere_free_energy.sum((1, 2)) * Mm_per_pixel ** 2
subframe_free_energy_h = subframe_free_energy.sum((1, 2)) * Mm_per_pixel ** 2
ax.plot(cartesian_free_energy_h, height, label='Cartesian')
ax.plot(spherical_free_energy_h, height, label='Spherical')
ax.plot(subframe_free_energy_h, height, label='Subframe')
ax.plot(pb_energy, height, label='Potential Boundary')
ax.set_ylabel('Height [Mm]')
ax.set_xlabel('Free Energy [erg/Mm]')
ax.set_ylim(0, 100)
ax.semilogx()

ax = axs[3]
cartesian_jxb = np.linalg.norm(np.cross(j_cartesian, b_cartesian), axis=-1).mean((0, 1))
spherical_jxb = np.linalg.norm(np.cross(j_spherical, b_spherical), axis=-1).mean((1, 2))
subframe_jxb = np.linalg.norm(np.cross(j_subframe, b_subframe), axis=-1).mean((1, 2))
pb_jxb = np.linalg.norm(np.cross(j_pb, b_pb), axis=-1).mean((1, 2))
ax.plot(cartesian_jxb, height, label='Cartesian')
ax.plot(spherical_jxb, height, label='Spherical')
ax.plot(subframe_jxb, height, label='Subframe')
ax.plot(pb_jxb, height, label='Potential Boundary', linestyle='--')
ax.set_ylabel('Height [Mm]')
ax.set_xlabel('|JxB| [G$^2$/Mm]')
ax.set_ylim(0, 100)
ax.semilogx()


ax = axs[4]
ax.plot(energy_gradient_cartesian.sum((0, 1)), height, label='Cartesian')
ax.plot(energy_gradient_spherical.sum((1, 2)), height, label='Spherical')
ax.plot(energy_gradient_subframe.sum((1, 2)), height, label='Subframe')
ax.plot(energy_gradient_pb.sum((1, 2)), height, label='Potential Boundary')
ax.set_ylabel('Height [Mm]')
ax.set_xlabel('Energy Gradient [erg/Mm]')
ax.set_ylim(0, 100)
ax.set_xlim(-2, 10)
ax.axvline(0, color='k', linestyle='--')

axs[-1].legend(fontsize=16)
fig.tight_layout()
plt.savefig('/glade/work/rjarolim/nf2/spherical/evaluation/height.png', dpi=300, transparent=True)
plt.close()
