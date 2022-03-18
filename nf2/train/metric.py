import numpy as np


def divergence(b_field):  # (x, y, z, (xyz-field))
    div_B = np.stack([np.gradient(b_field[..., i], axis=i, edge_order=2) for i in range(3)], axis=-1).sum(-1)
    return div_B


def curl(b_field):  # (x, y, z)
    _, dFx_dy, dFx_dz = np.gradient(b_field[..., 0], axis=[0, 1, 2], edge_order=2)
    dFy_dx, _, dFy_dz = np.gradient(b_field[..., 1], axis=[0, 1, 2], edge_order=2)
    dFz_dx, dFz_dy, _ = np.gradient(b_field[..., 2], axis=[0, 1, 2], edge_order=2)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    return np.stack([rot_x, rot_y, rot_z], -1)


def lorentz_force(b_field, j_field=None):
    j_field = j_field if j_field is not None else curl(b_field)
    l = np.cross(j_field, b_field, axis=-1)
    return l


def vector_norm(vector):
    return np.sqrt((vector ** 2).sum(-1))


def angle(b_field, j_field):
    norm = vector_norm(b_field) * vector_norm(j_field) + 1e-7
    j_cross_b = np.cross(j_field, b_field, axis=-1)
    sig = vector_norm(j_cross_b) / norm
    return np.arcsin(np.clip(sig, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)


def normalized_divergence(b_field):
    return np.abs(divergence(b_field)) / (vector_norm(b_field) + 1e-7)


def weighted_sigma(b, j=None):
    j = j if j is not None else curl(b)
    sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma) * (180 / np.pi)
    return theta_j


def energy(b):
    return (b ** 2).sum(-1) / (8 * np.pi)
