import numpy as np
from astropy.nddata import block_reduce

from nf2.loader.base import SlicesDataModule


class NumpyDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, bin=1, use_bz=False, components=False, *args, **kwargs):
        b_slices = np.load(data_path)
        if slices:
            b_slices = b_slices[:, :, slices]
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        if use_bz:
            b_slices[:, :, 1:, 0] = None
            b_slices[:, :, 1:, 1] = None
        if components:
            for i, c in enumerate(components):
                filter = [i for i in [0, 1, 2] if i not in c]
                b_slices[:, :, i, filter] = None
        super().__init__(b_slices, *args, **kwargs)
