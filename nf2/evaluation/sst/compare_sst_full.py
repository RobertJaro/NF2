import argparse
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nf2.evaluation.metric import energy, theta_J
from nf2.evaluation.output import CartesianOutput, HeightTransformOutput

# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluate SST extrapolation.')
parser.add_argument('--output', type=str, help='output path.')
args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/sst/evaluation'
os.makedirs(out_path, exist_ok=True)

# select intensity data in wings
sst_data = fits.getdata('/glade/work/rjarolim/data/SST/panorama_8542_StkI.fits')[10]

config1_file = '/glade/work/rjarolim/nf2/sst/13392_1slices_0851_v01/extrapolation_result.nf2'
config2_file = '/glade/work/rjarolim/nf2/sst/13392_7699_0851_v01/extrapolation_result.nf2'
config3_file = '/glade/work/rjarolim/nf2/sst/13392_2slices_0851_v02/extrapolation_result.nf2'
config4_file = '/glade/work/rjarolim/nf2/sst/13392_3slices_0851_v04/extrapolation_result.nf2'
config5_file = '/glade/work/rjarolim/nf2/sst/13392_4slices_0851_v01/extrapolation_result.nf2'

config1_model = CartesianOutput(config1_file)

config2_model = CartesianOutput(config2_file)

config3_model = CartesianOutput(config3_file)

config4_model = CartesianOutput(config4_file)

config5_model = CartesianOutput(config5_file)


config1_height_out = None

config2_height_out = None

config3_height_model = HeightTransformOutput(config3_file)
config3_height_out = config3_height_model.load_height_mapping()

config4_height_model = HeightTransformOutput(config4_file)
config4_height_out = config4_height_model.load_height_mapping()

config5_height_model = HeightTransformOutput(config5_file)
config5_height_out = config5_height_model.load_height_mapping()

x_min, x_max = config1_model.coord_range[0]
y_min, y_max = config1_model.coord_range[1]

Mm_per_pixel = 0.72
height_range = [0, 40]
Mm_per_ds = config1_model.Mm_per_ds
ds_per_pixel = Mm_per_pixel / Mm_per_ds

config1_out = config1_model.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range,
                                      metrics=['j', 'b_nabla_bz', 'free_energy'], progress=True)
config2_out = config2_model.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range,
                                      metrics=['j', 'b_nabla_bz', 'free_energy'], progress=True)
config3_out = config3_model.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range,
                                      metrics=['j', 'b_nabla_bz', 'free_energy'], progress=True)
config4_out = config4_model.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range,
                                      metrics=['j', 'b_nabla_bz', 'free_energy'], progress=True)
config5_out = config5_model.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range,
                                      metrics=['j', 'b_nabla_bz', 'free_energy'], progress=True)



config_outs = [config1_out, config2_out, config3_out, config4_out, config5_out]
config_height_outs = [config1_height_out, config2_height_out, config3_height_out, config4_height_out, config5_height_out]
labels = ['config1', 'config2', 'config3', 'config4', 'config5']
# axs[0, 0].set_title('6173 Å')
# axs[0, 1].set_title('+ 8542 Å')
# axs[0, 2].set_title(r'+ 7699 Å')
# axs[0, 3].set_title(r'+ 5896 Å ($B_\text{LOS}$)')
########################################################################################################################
y_slice = 150 // 2  # int(50 // Mm_per_pixel)
y_range = [45, 85]
j_norm = LogNorm(vmin=1e11, vmax=1e13)
b_norm = LogNorm(vmin=10, vmax=1e3)
extent = np.array([x_min, x_max, y_min, y_max]) * Mm_per_ds
yz_extent = [y_min * Mm_per_ds, y_max * Mm_per_ds, *height_range]

fig, axs = plt.subplots(4, 5, figsize=(20, 10))


for i, (model_out, height_out) in enumerate(zip(config_outs, config_height_outs)):
    column = axs[:, i]
    #
    ax = column[0]
    im = ax.imshow(model_out['b'][:, :, 0, 2].to_value(u.G).T, origin='lower', cmap='gray',
                   vmin=-2000, vmax=2000, extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')
    ax.plot([y_slice * Mm_per_pixel, y_slice * Mm_per_pixel], [y_range[0], y_range[1]], color='red', linestyle='--')
    #
    ax = column[1]
    j_mag = np.linalg.norm(model_out['metrics']['j'].to_value(u.G / u.s), axis=-1)
    im = ax.imshow(j_mag.sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=j_norm, cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\int{|J|}\,dz$ [G cm s$^{-1}$]')
    #
    ax = column[2]
    im = ax.imshow(model_out['metrics']['b_nabla_bz'][y_slice, :, :].T, origin='lower',
                   cmap='RdBu_r', vmin=-.1, vmax=.1, extent=yz_extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot  \nabla \hat{B}_z$ [G Mm$^{-1}$]')
    ax.set_xlim(y_range)
    ax.set_ylim([0, 30])
    #
    if height_out is not None:
        for height_mapping in height_out:
            h_coords = height_mapping['coords']
            h_y_slice = np.argmin(np.abs(h_coords[:, 0, 0, 0].to_value(u.Mm) - y_slice * Mm_per_pixel))
            ax.plot(np.linspace(y_min, y_max, h_coords.shape[1]) * Mm_per_ds, h_coords[h_y_slice, :, 0, 2].to_value(u.Mm),
                    linestyle='--', color='black')
    #
    ax = column[3]
    b_mag = np.linalg.norm(model_out['b'].to_value(u.G), axis=-1)
    im = ax.imshow(b_mag[y_slice, :, :].T, origin='lower', extent=yz_extent, norm=b_norm, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|B|$ [G]')
    ax.set_xlim(y_range)
    ax.set_ylim([0, 30])

for i, label in enumerate(labels):
    axs[0, i].set_title(label)


plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_full_comparison.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################
# plot energy and current density profiles

fig, axs = plt.subplots(1, 4, figsize=(10, 10))


heights = np.linspace(0, 40, config1_out['b'].shape[2])
for i, model_out in enumerate(config_outs):
    energy_profile = energy(model_out['b']).sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2
    free_energy_profile = model_out['metrics']['free_energy'].sum((0, 1)) * (Mm_per_pixel * 1e8) ** 2
    current_profile = np.linalg.norm(model_out['metrics']['j'].to_value(u.G / u.s), axis=-1).sum((0, 1)) * (
            Mm_per_pixel * 1e8) ** 2
    b_mag = np.linalg.norm(model_out['b'].to_value(u.G), axis=-1)
    jxb = np.linalg.norm(
        np.cross(model_out['b'].to_value(u.G), model_out['metrics']['j'].to_value(u.G / u.s), axis=-1), axis=-1)
    jxb_profile = (jxb / b_mag).sum((0, 1))
    #
    axs[0].plot(energy_profile, heights, label=labels[i])
    axs[1].plot(free_energy_profile, heights, label=labels[i])
    axs[2].plot(current_profile, heights, label=labels[i])
    axs[3].plot(jxb_profile, heights, label=labels[i])

ax = axs[0]
ax.set_xlabel('Energy [erg cm$^{-1}$]')
ax.set_ylabel('Height [Mm]')
ax.set_title('Energy')
ax.legend()
ax.grid()
ax.set_ylim([0, 40])

ax = axs[1]
ax.set_xlabel('Free energy [erg cm$^{-1}$]')
ax.set_ylabel('Height [Mm]')
ax.set_title('Free energy')
ax.legend()
ax.grid()
ax.set_ylim([0, 40])

ax = axs[2]
ax.set_xlabel('Current density [G cm$^{2}$ s$^{-1}$]')
ax.set_ylabel('Height [Mm]')
ax.set_title('Current density')
ax.legend()
ax.grid()
ax.set_ylim([0, 40])

ax = axs[3]
ax.set_xlabel('(J x B) / B [G cm$^{-1}$ s$^{-1}$]')
ax.set_ylabel('Height [Mm]')
ax.set_title('Force-freeness')
ax.legend()
ax.grid()
ax.set_ylim([0, 40])

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'sst_full_profiles.png'), dpi=300, transparent=True)
plt.close()

########################################################################################################################
# print total energy stored in the field

for config_out in config_outs:
    energy_total = energy(config_out['b'].to_value(u.G)).sum() * (Mm_per_pixel * 1e8) ** 3
    print(f'Energy: {energy_total * 1e-32:.2f} 10^32 erg')

for config_out in config_outs:
    free_energy_total = config_out['metrics']['free_energy'].sum() * (Mm_per_pixel * 1e8 * u.cm) ** 3
    print(f'Free Energy: {free_energy_total.to_value(u.erg) * 1e-32:.2f} 10^32 erg')

########################################################################################################################
# print theta_J

for config_out in config_outs:
    theta = theta_J(config_out['b'].to_value(u.G), config_out['metrics']['j'].to_value(u.G / u.s))
    print(f'Theta_J: {theta:.2f}')


########################################################################################################################
# plot height mapping

extent = np.array([x_min, x_max, y_min, y_max]) * Mm_per_ds

height_model = HeightTransformOutput(config5_file)
height_out = height_model.load_height_mapping()

fig, axs = plt.subplots(len(height_out) + 2, 1, figsize=(10, 15))

j_mag = np.linalg.norm(config5_out['metrics']['j'].to_value(u.G / u.s), axis=-1)
axs[0].imshow(sst_data, origin='lower', extent=extent, cmap='gray')
axs[0].contour(j_mag.sum(2).T * Mm_per_pixel * 1e8, extent=extent, levels=[1e12], colors=['red'], linewidths=0.5)
axs[0].grid()

axs[1].imshow(j_mag.sum(2).T * Mm_per_pixel * 1e8, origin='lower', extent=extent, norm=LogNorm(1e11, 1e13),
              cmap='plasma')

for i, h_out in enumerate(height_out):
    ax = axs[i + 2]
    im = ax.imshow(h_out['coords'][..., 0, 2].to_value(u.Mm).T, origin='lower', extent=extent, cmap='jet',
                   norm=LogNorm(1e-2, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='Height [Mm]')
    ax.grid()

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'height_mapping.png'), dpi=300, transparent=True)
plt.close()
