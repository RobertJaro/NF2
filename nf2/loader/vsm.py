import numpy as np

from nf2.loader.base import SlicesDataModule


class VSMDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, *args, **kwargs):
        dict_data = np.load(data_path, allow_pickle=True)
        sharp_cube = dict_data.item().get('sharp')
        vsm_cube = dict_data.item().get('vsm')
        vsm_cube = np.stack([np.ones_like(vsm_cube) * np.nan, np.ones_like(vsm_cube) * np.nan, vsm_cube])
        b_slices = np.stack([sharp_cube, vsm_cube], 1).T
        if slices is not None:
            b_slices = b_slices[:, :, slices]

        super().__init__(b_slices, *args, **kwargs)
