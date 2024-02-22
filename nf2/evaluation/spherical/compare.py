import numpy as np
from sunpy.coordinates import frames
from sunpy.map import Map

from astropy import units as u

from nf2.evaluation.metric import divergence, weighted_theta, theta_J
from nf2.evaluation.output import CartesianOutput, SphericalOutput
from nf2.evaluation.vtk import save_vtk

cartesian_model = CartesianOutput('/glade/work/rjarolim/nf2/cartesian/sharp_377_v03/extrapolation_result.nf2')
cartesian_out = cartesian_model.load_cube(progress=True, Mm_per_pixel=0.72)

reference_map = Map('/glade/work/rjarolim/data/nf2/377/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits')

bottom_left = reference_map.bottom_left_coord.transform_to(frames.HeliographicCarrington)
top_right = reference_map.top_right_coord.transform_to(frames.HeliographicCarrington)

spherical_model = SphericalOutput('/glade/work/rjarolim/nf2/spherical/377_v01/extrapolation_result.nf2')
Mm_per_pixel = cartesian_out['Mm_per_pixel'] * (u.Mm / u.pix)
spherical_out = spherical_model.load(latitude_range=((np.pi / 2 * u.rad) - top_right.lat, (np.pi / 2 * u.rad) - bottom_left.lat),
                                     longitude_range=(bottom_left.lon, top_right.lon),
                                     radius_range=(1 * u.solRad, 1 * u.solRad + 100 * u.Mm), resolution=1 / Mm_per_pixel,
                                     progress=True)

b_spherical = spherical_out['B'].value
b_cartesian = cartesian_out['B'].value

j_spherical = spherical_out['J'].value
j_cartesian = cartesian_out['J'].value

coords_spherical = spherical_out['coords']
coords_cartesian = cartesian_out['coords']

div_spherical = np.nanmean(np.abs(divergence(b_spherical)))
div_cartesian = np.nanmean(np.abs(divergence(b_cartesian)))

theta_spherical = theta_J(b_spherical, j_spherical)
theta_cartesian = theta_J(b_cartesian, j_cartesian)

print(f'Divergence: Spherical: {div_spherical:.2e}, Cartesian: {div_cartesian:.2e}')
print(f'Weighted Theta: Spherical: {theta_spherical:.2f}, Cartesian: {theta_cartesian:.2f}')

print(f'Saving Cube (spherical): {coords_spherical.shape}')
save_vtk('/glade/work/rjarolim/nf2/spherical/evaluation/spherical.vtk', coords_spherical, vectors={'B': np.nan_to_num(b_spherical), 'J': np.nan_to_num(j_spherical)}, Mm_per_pix=0.72)
print(f'Saving Cube (cartesian): {coords_cartesian.shape}')
save_vtk('/glade/work/rjarolim/nf2/spherical/evaluation/cartesian.vtk', coords_cartesian, vectors={'B': b_cartesian, 'J': j_cartesian}, Mm_per_pix=0.72)