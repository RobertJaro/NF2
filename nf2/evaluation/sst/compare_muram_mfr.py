import os

import numpy as np
from astropy import units as u, constants
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nf2.evaluation.output_metrics import squashing_factor, free_energy

from nf2.evaluation.metric import b_nabla_bz, curl, energy
from nf2.evaluation.output import CartesianOutput, HeightTransformOutput
from nf2.loader.muram import MURaMSnapshot

# if __name__ == '__main__':
# parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
# parser.add_argument('--output', type=str, help='output path.')
# args = parser.parse_args()

out_path = '/glade/work/rjarolim/nf2/topology/results'
os.makedirs(out_path, exist_ok=True)

Mm_per_pixel = 0.192  # 0.192
height = 10
# x_slice = int(32 / Mm_per_pixel)  # 40 Mm


mfr_muram_snapshot = MURaMSnapshot('/glade/campaign/hao/radmhd/Rempel/Spot_Motion/case_B/3D', iteration=474000)
mfr_muram_cube = mfr_muram_snapshot.load_cube(resolution=Mm_per_pixel * u.Mm / u.pix, height=height * u.Mm, method='min')
mfr_muram_b = mfr_muram_cube['B']
mfr_muram_BnablaBz = b_nabla_bz(mfr_muram_b) / Mm_per_pixel
mfr_muram_j = (curl(mfr_muram_b) / Mm_per_pixel * u.G / u.Mm) * constants.c / (4 * np.pi)
mfr_muram_j = mfr_muram_j.to_value(u.G / u.s)

nf2_1slice = '/glade/work/rjarolim/nf2/topology/muram_mfr_1slices_ambiguous_v01/extrapolation_result.nf2'
nf2_2slice = '/glade/work/rjarolim/nf2/topology/muram_mfr_2slices_ambiguous_v01/extrapolation_result.nf2'
nf2_3slice = '/glade/work/rjarolim/nf2/topology/muram_mfr_3slices_ambiguous_v01/extrapolation_result.nf2'

model_1slice = CartesianOutput(nf2_1slice)
model_2slice = CartesianOutput(nf2_2slice)
model_3slice = CartesianOutput(nf2_3slice)

height_2slices_model = HeightTransformOutput(nf2_2slice)
height_2slices_out = height_2slices_model.load_height_mapping()

height_2slices_amb_model = HeightTransformOutput(nf2_3slice)
height_2slices_amb_out = height_2slices_amb_model.load_height_mapping()

muram_heights = []
for tau in [1e-4, 1e-6]:
    height_muram = mfr_muram_snapshot.load_tau_height(tau, method='min')
    print(
        f'MURAM Height {tau:.1e}: {height_muram.to(u.Mm).min():.2f} -- {height_muram.to(u.Mm).max():.2f}; {height_muram.to(u.Mm).mean():.2f}')
    muram_heights.append(height_muram)

Mm_per_ds = model_1slice.Mm_per_ds
ds_per_pixel = Mm_per_pixel / Mm_per_ds

x_min, x_max = model_1slice.coord_range[0] * model_1slice.Mm_per_ds
y_min, y_max = model_1slice.coord_range[1] * model_1slice.Mm_per_ds

out_1slice = model_1slice.load_cube([0, height], metrics=['j', 'b_nabla_bz'], progress=True,
                                    Mm_per_pixel=Mm_per_pixel)
out_2slice = model_2slice.load_cube([0, height], metrics=['j', 'b_nabla_bz'], progress=True,
                                    Mm_per_pixel=Mm_per_pixel)
out_3slice = model_3slice.load_cube([0, height], metrics=['j', 'b_nabla_bz'], progress=True,
                                    Mm_per_pixel=Mm_per_pixel)

out_1slice['metrics']['b_nabla_bz'] = b_nabla_bz(out_1slice['b']) / Mm_per_pixel
out_2slice['metrics']['b_nabla_bz'] = b_nabla_bz(out_2slice['b']) / Mm_per_pixel
out_3slice['metrics']['b_nabla_bz'] = b_nabla_bz(out_3slice['b']) / Mm_per_pixel


# compute twist and squashing factor
offset = int(2 / Mm_per_pixel)
mfr_muram_squashing_out = squashing_factor(mfr_muram_b[:, :, offset:])
mfr_1slice_squashing_out = squashing_factor(out_1slice['b'][:, :, offset:])
mfr_2slice_squashing_out = squashing_factor(out_2slice['b'][:, :, offset:])
mfr_2slices_amb_squashing_out = squashing_factor(out_3slice['b'][:, :, offset:])

x_slice = 33 + x_min  # Mm
# x_slice = 37 + x_min  # Mm
# x_slice = 40 + x_min  # Mm
xlim = [2, 12]
# get pixel index close to x_slice in Mm
x_slice_pix = np.argmin(np.abs(np.linspace(x_min, x_max, mfr_muram_b.shape[0]) - x_slice))
print(f'x_slice_pix: {x_slice_pix}, x_slice: {x_slice} Mm')

# compute free magnetic energy
mfr_muram_free_energy = free_energy(mfr_muram_b * u.G)  # erg cm^-3
mfr_1slice_free_energy = free_energy(out_1slice['b'])  # erg cm^-3
mfr_2slice_free_energy = free_energy(out_2slice['b'])  # erg cm^-3
mfr_2slices_amb_free_energy = free_energy(out_3slice['b'])  # erg cm^-3

##############################################################################
######################### plot Bz
extent = np.array([x_min, x_max, y_min, y_max])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

im = ax.imshow(mfr_muram_b[:, :, 0, 2].T, origin='lower', cmap='gray',
               vmin=-1000, vmax=1000, extent=extent)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label='Bz [G]')

# draw red line at slice position
ax.axvline(x_slice, color='red', linestyle='--')

plt.savefig(os.path.join(out_path, 'bz.png'))
plt.close()

##############################################################################
######################### plot b_nabla_bz


extent = [y_min, y_max, 0, height]


def _plot(ax, b, coord, bnbz, title, heights=[]):
    c = coord[x_slice_pix, :, :, 1:]
    #
    im = ax.imshow(bnbz[x_slice_pix, :, :].T, origin='lower', cmap='coolwarm', vmin=-.1, vmax=.1, extent=extent)
    ax.set_title(title)
    #
    byz_pre = b[x_slice_pix, :, :, 1:]
    coord_q = c#[::2, ::2]  # block_reduce(coord, (3, 3, 1), np.mean)
    b_q = byz_pre#[::2, ::2]  # block_reduce(byz_pre, (3, 3, 1), np.mean)
    b_q = b_q / np.linalg.norm(b_q, axis=-1, keepdims=True)
    ax.quiver(coord_q[..., 0], coord_q[..., 1], b_q[..., 0], b_q[..., 1], color='darkgray', scale=40, pivot='middle')
    #
    for h_coords in heights:
        ax.plot(np.linspace(y_min, y_max, h_coords.shape[1]), h_coords[x_slice_pix, :].to_value(u.Mm),
                color='black', linestyle='--')
    ax.set_ylim([0, height])
    # horizontal line at z=5 Mm
    ax.axhline(5.0, color='red', linestyle=':')
    #
    return im


fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

coord = np.stack(np.meshgrid(
    np.linspace(x_min, x_max, mfr_muram_b.shape[0]),
    np.linspace(y_min, y_max, mfr_muram_b.shape[1]),
    np.linspace(0, height, mfr_muram_b.shape[2]), indexing='ij'), -1)
_plot(axs[0], mfr_muram_b, coord, mfr_muram_BnablaBz, 'MURaM', heights=muram_heights)

coord = out_1slice['coords']
_plot(axs[1], out_1slice['b'].to_value(u.G), coord, out_1slice['metrics']['b_nabla_bz'], 'NF2 - single height')

_plot(axs[2], out_2slice['b'].to_value(u.G), coord, out_2slice['metrics']['b_nabla_bz'], 'NF2 - multi height',
      heights=[height_2slices_out[i]['coords'][:, :, 0, 2] for i in range(len(height_2slices_out))])

im = _plot(axs[3], out_3slice['b'].to_value(u.G), coord, out_3slice['metrics']['b_nabla_bz'],
           'NF2 - multi height (ambiguous)',
           heights=[height_2slices_amb_out[i]['coords'][:, :, 0, 2] for i in range(len(height_2slices_amb_out))])

divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\hat{B} \cdot \nabla \hat{B}_z$ [Mm$^{-1}$]')

[ax.set_xlabel('Y [Mm]') for ax in axs]
axs[0].set_ylabel('Z [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

[ax.set_xlim(xlim) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'b_nabla_bz.png'), dpi=300, transparent=True)
plt.close()

##############################################################################
######################### plot energy

extent = [y_min, y_max, 0, height]
b_norm = LogNorm(vmin=1e2, vmax=1e5)

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(energy(mfr_muram_b[x_slice_pix, :, :]).T, origin='lower', extent=extent, norm=b_norm, cmap='jet')
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(energy(out_1slice['b'][x_slice_pix, :, :].to_value(u.G)).T, origin='lower', extent=extent,
          norm=b_norm, cmap='jet')
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(energy(out_2slice['b'][x_slice_pix, :, :].to_value(u.G)).T, origin='lower', extent=extent,
          norm=b_norm, cmap='jet')
ax.set_title('NF2 - multi height')

ax = axs[3]
ax.imshow(energy(out_3slice['b'][x_slice_pix, :, :].to_value(u.G)).T, origin='lower',
          extent=extent, norm=b_norm, cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'Magnetic Energy [ergs cm$^{-3}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

[ax.set_xlim(xlim) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'energy.png'), dpi=300, transparent=True)
plt.close()

##############################################################################
######################### plot j
z_height = int(2 / Mm_per_pixel)

extent = [x_min, x_max, y_min, y_max]
j_norm = LogNorm()

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(np.linalg.norm(mfr_muram_j[:, :, z_height:], axis=-1).sum(2).T * Mm_per_pixel * 1e8, origin='lower',
               extent=extent,
               norm=j_norm)
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(np.linalg.norm(out_1slice['metrics']['j'][:, :, z_height:].to_value(u.G / u.s), axis=-1).sum(
    2).T * Mm_per_pixel * 1e8,
          origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(np.linalg.norm(out_2slice['metrics']['j'][:, :, z_height:].to_value(u.G / u.s), axis=-1).sum(
    2).T * Mm_per_pixel * 1e8,
          origin='lower', extent=extent, norm=j_norm)
ax.set_title('NF2 - multi height')

ax = axs[3]
ax.imshow(
    np.linalg.norm(out_3slice['metrics']['j'][:, :, z_height:].to_value(u.G / u.s), axis=-1).sum(
        2).T * Mm_per_pixel * 1e8,
    origin='lower', extent=extent, norm=j_norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$|J|$ [G cm s$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'j.png'), dpi=300, transparent=True)
plt.close()


##############################################################################
######################### plot alpha

def _alpha(b):
    j = curl(b) / Mm_per_pixel
    return (b * j).sum(-1) / (np.linalg.norm(b, axis=-1) ** 2 + 1e-6)


extent = [y_min, y_max, 0, height]
alpha_norm = Normalize(-2, 2)

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
muram_alpha = _alpha(mfr_muram_b)
im = ax.imshow(muram_alpha[x_slice_pix, :, :].T, origin='lower', extent=extent, norm=alpha_norm, cmap='seismic')
ax.set_title('MURaM')

ax = axs[1]
slice1_alpha = _alpha(out_1slice['b'].to_value(u.G))
ax.imshow(slice1_alpha[x_slice_pix, :, :].T, origin='lower', extent=extent,
          norm=alpha_norm, cmap='seismic')
ax.set_title('NF2 - single height')

ax = axs[2]
slice2_alpha = _alpha(out_2slice['b'].to_value(u.G))
ax.imshow(slice2_alpha[x_slice_pix, :, :].T, origin='lower', extent=extent,
          norm=alpha_norm, cmap='seismic')
ax.set_title('NF2 - multi height')

ax = axs[3]
slice2_amb_alpha = _alpha(out_3slice['b'].to_value(u.G))
ax.imshow(slice2_amb_alpha[x_slice_pix, :, :].T, origin='lower', extent=extent, norm=alpha_norm, cmap='seismic')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'$\alpha$ [Mm$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

[ax.set_xlim(xlim) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'alpha.png'), dpi=300, transparent=True)
plt.close()

##############################################################################
######################### plot twist

twist_norm = Normalize(-2, 2)

extent = [y_min, y_max, offset * Mm_per_pixel, height]

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(mfr_muram_squashing_out['twist'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=twist_norm, cmap='seismic')
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(mfr_1slice_squashing_out['twist'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=twist_norm, cmap='seismic')
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(mfr_2slice_squashing_out['twist'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=twist_norm, cmap='seismic')
ax.set_title('NF2 - multi height')

ax = axs[3]
ax.imshow(mfr_2slices_amb_squashing_out['twist'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=twist_norm, cmap='seismic')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'Twist [turns Mm$^{-1}$]')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]

[ax.set_xlim(xlim) for ax in axs]

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'twist.png'), dpi=300, transparent=True)
plt.close()


##############################################################################
######################### plot squashing factor

squashing_norm = LogNorm(vmin=1, vmax=1e3)
extent = [y_min, y_max, offset * Mm_per_pixel, height]

fig, axs = plt.subplots(1, 4, figsize=(15, 2.7))

ax = axs[0]
im = ax.imshow(mfr_muram_squashing_out['q'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=squashing_norm, cmap='viridis')
ax.set_title('MURaM')

ax = axs[1]
ax.imshow(mfr_1slice_squashing_out['q'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=squashing_norm, cmap='viridis')
ax.set_title('NF2 - single height')

ax = axs[2]
ax.imshow(mfr_2slice_squashing_out['q'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=squashing_norm, cmap='viridis')
ax.set_title('NF2 - multi height')

ax = axs[3]
ax.imshow(mfr_2slices_amb_squashing_out['q'][x_slice_pix, :, :].T, origin='lower', extent=extent, norm=squashing_norm, cmap='viridis')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical', label=r'Squashing Factor Q')
ax.set_title('NF2 - multi height (ambiguous)')

[ax.set_xlabel('X [Mm]') for ax in axs]
axs[0].set_ylabel('Y [Mm]')
[ax.axes.yaxis.set_ticklabels([]) for ax in axs[1:]]
[ax.set_xlim(xlim) for ax in axs]
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'squashing_factor.png'), dpi=300,
            transparent=True)
plt.close()

##############################################################################
######################### plot energy comparison

z_coords = out_1slice['coords'][0, 0, :, 2]
muram_z_coords = np.linspace(0, mfr_muram_b.shape[2], mfr_muram_b.shape[2]) * Mm_per_pixel
ds3 = (Mm_per_pixel * u.Mm).to_value(u.cm) ** 3

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(energy(mfr_muram_b).sum((0, 1)) * ds3, muram_z_coords, label='MURaM', color='black')
ax.plot(energy(out_1slice['b'].to_value(u.G)).sum((0, 1)) * ds3, z_coords, label='NF2 - single height', color='blue')
ax.plot(energy(out_2slice['b'].to_value(u.G)).sum((0, 1)) * ds3, z_coords, label='NF2 - multi height', color='orange')
ax.plot(energy(out_3slice['b'].to_value(u.G)).sum((0, 1)) * ds3, z_coords, label='NF2 - multi height (ambiguous)', color='green')
ax.set_xlabel('Magnetic Energy [ergs]')
ax.set_ylabel('Height [Mm]')
ax.set_yscale('linear')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'energy_comparison.png'), dpi=300, transparent=True)
plt.close()

##################################################################
###################### compute B grad alpha

def b_grad_alpha(b, alpha):  # (x, y, z)
    dAlpha_dx, dAlpha_dy, dAlpha_dz = np.gradient(alpha, axis=[0, 1, 2], edge_order=2)
    grad_alpha = np.stack([dAlpha_dx, dAlpha_dy, dAlpha_dz], axis=-1)
    b_unit = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (b_unit * grad_alpha).sum(-1)


muram_b_grad_alpha = b_grad_alpha(mfr_muram_b, muram_alpha)
slice1_b_grad_alpha = b_grad_alpha(out_1slice['b'].to_value(u.G), slice1_alpha)
slice2_b_grad_alpha = b_grad_alpha(out_2slice['b'].to_value(u.G), slice2_alpha)
slice2_amb_b_grad_alpha = b_grad_alpha(out_3slice['b'].to_value(u.G), slice2_amb_alpha)

print('B_grad_alpha')
print('MURaM', np.abs(muram_b_grad_alpha).mean())
print('1 Slice', np.abs(slice1_b_grad_alpha).mean())
print('2 Slices', np.abs(slice2_b_grad_alpha).mean())
print('2 Slices Amb', np.abs(slice2_amb_b_grad_alpha).mean())
