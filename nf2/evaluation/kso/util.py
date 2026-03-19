import numpy as np
from sunpy.map import Map


def load_KSO(f):
    """
    Load a KSO H-alpha FITS file and return a SunPy Map.
    """
    kso_map = Map(f)
    angle = -kso_map.meta["angle"]

    kso_map.meta["waveunit"] = "AA"
    kso_map.meta["arcs_pp"] = kso_map.scale[0].value

    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))

    kso_map.meta["PC1_1"] = c
    kso_map.meta["PC1_2"] = -s
    kso_map.meta["PC2_1"] = s
    kso_map.meta["PC2_2"] = c
    return kso_map