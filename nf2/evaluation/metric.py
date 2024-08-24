import numpy as np

from nf2.potential.potential_field import get_potential_field


def divergence(b_field):  # (x, y, z, (xyz-field))
    div_B = np.stack([np.gradient(b_field[..., i], axis=i, edge_order=2) for i in range(3)], axis=-1).sum(-1)
    return div_B

def divergence_jacobian(jac_matrix):
    dBx_dx = jac_matrix[..., 0, 0]
    dBy_dy = jac_matrix[..., 1, 1]
    dBz_dz = jac_matrix[..., 2, 2]
    div_B = dBx_dx + dBy_dy + dBz_dz
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


def weighted_theta(b, j=None):
    j = j if j is not None else curl(b)
    sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma)
    theta_j = np.rad2deg(theta_j)
    return theta_j

def theta_J(b, j=None):
    j = j if j is not None else curl(b)
    norm = vector_norm(b) * vector_norm(j) + 1e-7
    sigma = vector_norm(lorentz_force(b, j)) / norm
    j_weight = vector_norm(j)
    angle = np.nansum(sigma * j_weight) / np.nansum(j_weight)
    angle = np.clip(angle, -1. + 1e-7, 1. - 1e-7)
    t_angle = np.arcsin(angle)
    t_angle = np.rad2deg(t_angle)
    return t_angle

def sigma_J(b, j):
    return (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / (vector_norm(j).sum() + 1e-6)


def energy(b):
    return (b ** 2).sum(-1) / (8 * np.pi)


def evaluate(b, B):
    result = {}
    result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
    M = np.prod(B.shape[:-1])
    result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

    result['E_n'] = vector_norm(b - B).sum() / vector_norm(B).sum()

    result['E_m'] = 1 / M * (vector_norm(b - B) / vector_norm(B)).sum()

    result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

    # B_potential = get_potential_field(B[:, :, 0, 2], 64)
    #
    # result['eps_p'] = (vector_norm(b[:, :, :64]) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()
    # result['eps_p_ll'] = (vector_norm(B[:, :, :64]) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

    j = curl(b)
    result['sig_J'] = sigma_J(b, j) * 1e2
    J = curl(B)
    result['sig_J_ll'] = sigma_J(B, J) * 1e2

    result['L1'] = (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()
    result['L2'] = (divergence(b) ** 2).mean()

    result['L1_B'] = (vector_norm(np.cross(curl(B), B, -1)) ** 2 / vector_norm(B) ** 2).mean()
    result['L2_B'] = (divergence(B) ** 2).mean()

    result['L2n'] = (np.abs(divergence(b)) / (vector_norm(b) + 1e-8)).mean() * 1e2
    result['L2n_B'] = (np.abs(divergence(B)) / (vector_norm(B) + 1e-8)).mean() * 1e2

    return result




def b_diff_error(b, B, B_error):
    b_err_diff = np.abs(b - B)
    b_err_diff = np.clip(b_err_diff, a_min=0, a_max=None) - B_error
    b_err_diff = np.linalg.norm(b_err_diff, axis=-1)
    return b_err_diff


def b_nabla_bz(b):
    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]

    jac_matrix = np.stack(np.gradient(b, axis=(0, 1, 2)), -1)
    dBx_dx = jac_matrix[..., 0, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dx = jac_matrix[..., 1, 0]
    dBy_dy = jac_matrix[..., 1, 1]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dx = jac_matrix[..., 2, 0]
    dBz_dy = jac_matrix[..., 2, 1]
    dBz_dz = jac_matrix[..., 2, 2]

    norm_B = np.linalg.norm(b, axis=-1)

    dnormB_dx = - norm_B ** -3 * (bx * dBx_dx + by * dBy_dx + bz * dBz_dx)
    dnormB_dy = - norm_B ** -3 * (bx * dBx_dy + by * dBy_dy + bz * dBz_dy)
    dnormB_dz = - norm_B ** -3 * (bx * dBx_dz + by * dBy_dz + bz * dBz_dz)

    b_nabla_bz = (bx * dBz_dx + by * dBz_dy + bz * dBz_dz) / norm_B ** 2 + \
                 (bz / norm_B) * (bx * dnormB_dx + by * dnormB_dy + bz * dnormB_dz)
    return b_nabla_bz