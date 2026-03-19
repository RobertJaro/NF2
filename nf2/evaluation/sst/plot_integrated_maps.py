import os

import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.colors import LogNorm

out_path = '/glade/work/rjarolim/nf2/topology/results/integrated_maps'
os.makedirs(out_path, exist_ok=True)

data_1slices = np.load('/glade/work/rjarolim/nf2/topology/results/13392_0851_1slices_v01.npz')
data_2slices = np.load('/glade/work/rjarolim/nf2/topology/results/13392_0851_2slices_v01.npz')

coords = data_1slices['coords']  # (x, y, z, 3)
Mm_per_pixel = data_1slices['Mm_per_pixel']  # scalar
m_per_pixel = Mm_per_pixel * 1e6
cm_per_pixel = m_per_pixel * 100

x_min, x_max = coords[:, :, :, 0].min(), coords[:, :, :, 0].max()
y_min, y_max = coords[:, :, :, 1].min(), coords[:, :, :, 1].max()
z_min, z_max = coords[:, :, :, 2].min(), coords[:, :, :, 2].max()

j_1slices = data_1slices['j']  # (x, y, z, 3)
j_2slices = data_2slices['j']  # (x, y, z, 3)
#
j_map_1slices = np.linalg.norm(j_1slices, axis=-1).sum(2) * m_per_pixel  # (x, y)
j_map_2slices = np.linalg.norm(j_2slices, axis=-1).sum(2) * m_per_pixel  # (x, y)

free_energy_1slices = data_1slices['free_energy']  # (x, y, z)
free_energy_2slices = data_2slices['free_energy']  # (x, y, z)
#
free_energy_map_1slices = free_energy_1slices.sum(2) * cm_per_pixel  # (x, y)
free_energy_map_2slices = free_energy_2slices.sum(2) * cm_per_pixel  # (x, y)

################################################
# create figure
extent = [x_min, x_max, y_min, y_max]
j_norm = LogNorm()
free_energy_norm = LogNorm()

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.05)

j_ax_1 = fig.add_subplot(gs[0, 0])
im0 = j_ax_1.imshow(j_map_1slices.T, origin='lower', cmap='plasma', extent=extent, norm=j_norm)
j_ax_1.set_xlabel('X [Mm]')
j_ax_1.set_ylabel('Y [Mm]')

j_ax_2 = fig.add_subplot(gs[0, 1])
im1 = j_ax_2.imshow(j_map_2slices.T, origin='lower', cmap='plasma', extent=extent, norm=j_norm)
j_ax_2.set_xlabel('X [Mm]')
j_ax_2.set_ylabel('Y [Mm]')

j_cax = fig.add_subplot(gs[0, 2])
cbar_j = fig.colorbar(im1, cax=j_cax)
cbar_j.set_label(r'$\sum_z |\mathbf{j}|$ [G$^2$ m / s]')

free_energy_ax_1 = fig.add_subplot(gs[1, 0])
im2 = free_energy_ax_1.imshow(free_energy_map_1slices.T, origin='lower', cmap='jet', extent=extent,
                              norm=free_energy_norm)
free_energy_ax_1.set_xlabel('X [Mm]')
free_energy_ax_1.set_ylabel('Y [Mm]')

free_energy_ax_2 = fig.add_subplot(gs[1, 1])
im3 = free_energy_ax_2.imshow(free_energy_map_2slices.T, origin='lower', cmap='jet', extent=extent,
                              norm=free_energy_norm)
free_energy_ax_2.set_xlabel('X [Mm]')
free_energy_ax_2.set_ylabel('Y [Mm]')

free_energy_cax = fig.add_subplot(gs[1, 2])
cbar_free_energy = fig.colorbar(im3, cax=free_energy_cax)
cbar_free_energy.set_label(r'$\sum_z E_{\rm free}$ [erg / cm$^2$]')

plt.savefig(os.path.join(out_path, 'integrated_maps.png'), dpi=300)
plt.close(fig)
