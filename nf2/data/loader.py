import numpy as np
from astropy.nddata import block_reduce

from nf2.potential.potential_field import get_potential_boundary

def load_potential_field_boundary(bz, height, reduce, only_top=False, pf_error=0.0, method='fft', **kwargs):
    if reduce > 1:
        bz = block_reduce(bz, (reduce, reduce), func=np.mean)
        height = height // reduce

    pf_coords, pf_values = get_potential_boundary(
        bz,
        height,
        only_top=only_top,
        method=method,
        **kwargs,
    )
    pf_values = np.array(pf_values, dtype=np.float32)
    pf_coords = np.array(pf_coords, dtype=np.float32) * reduce  # expand to original coordinate spacing
    pf_err = np.ones_like(pf_values) * pf_error
    return pf_coords, pf_err, pf_values
