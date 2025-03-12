import os.path

import numpy as np
from matplotlib import pyplot as plt

from nf2.evaluation.output import CartesianOutput
from astropy import units as u

base_path = "/glade/work/rjarolim/nf2/sharp/12760_vp_v01"
model = CartesianOutput("/glade/work/rjarolim/nf2/sharp/12760_vp_v01/extrapolation_result.nf2")
out = model.load_cube(progress=True, Mm_per_pixel=0.72)

b_norm = np.linalg.norm(out['b'], axis=-1).to_value(u.G)
h = out['coords'][0, 0, :, 2]
b_profile = b_norm.mean(axis=(0, 1))

b0 = b_profile[0]

log_fit = np.polyfit(h / 100, np.log(b_profile), 3, w=b_profile)
b_fit = np.exp(np.polyval(log_fit, h/ 100))
print('B0:', np.exp(log_fit[3]))
print(f'Fit: {log_fit[0]:.2e} h^3 + {log_fit[1]:.2e} h^2 + {log_fit[2]:.2e} h + {log_fit[3]:.2e}')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(b_profile, h, label='B')
ax.plot(b_fit, h, '--', label='fit')
ax.plot(b0 * np.exp(-h / 100 * 7), h, ':', label='custom')
ax.set_xlabel('B [G]')
ax.set_ylabel('Height [Mm]')
ax.set_ylim(0, None)
ax.legend()
fig.savefig(os.path.join(base_path, 'b_norm.png'))
plt.close(fig)