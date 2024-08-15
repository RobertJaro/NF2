import numpy as np


def spherical_to_cartesian_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = np.stack([
        np.stack([sin(t) * cos(p), cos(t) * cos(p), -sin(p)], -1),
        np.stack([sin(t) * sin(p), cos(t) * sin(p), cos(p)], -1),
        np.stack([cos(t), -sin(t), np.zeros_like(t)], -1)
    ], -2)
    #
    return matrix


def cartesian_to_spherical_matrix(c):
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    matrix = np.stack([
        np.stack([sin(t) * cos(p), sin(t) * sin(p), cos(t)], -1),
        np.stack([cos(t) * cos(p), cos(t) * sin(p), -sin(t)], -1),
        np.stack([-sin(p), cos(p), np.zeros_like(p)], -1)
    ], -2)
    #
    return matrix


def cartesian_rotation_matrix(theta, phi):
    m = np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), -np.sin(theta) * np.cos(phi)],
                  [np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                  [np.sin(theta), 0, np.cos(theta)]])
    return m


def vector_spherical_to_cartesian(v, c, f=np):
    vr, vt, vp = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return f.stack([vx, vy, vz], -1)


def vector_cartesian_to_spherical(v, c, f=np):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    r, t, p = c[..., 0], c[..., 1], c[..., 2]
    sin = f.sin
    cos = f.cos
    #
    vr = vx * sin(t) * cos(p) + vy * sin(t) * sin(p) + vz * cos(t)
    vt = vx * cos(t) * cos(p) + vy * cos(t) * sin(p) - vz * sin(t)
    vp = - vx * sin(p) + vy * cos(p)
    #
    return f.stack([vr, vt, vp], -1)


def spherical_to_cartesian(v, f=np):
    sin = f.sin
    cos = f.cos
    r, t, p = v[..., 0], v[..., 1], v[..., 2]
    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return f.stack([x, y, z], -1)


def cartesian_to_spherical(v, f=np):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    xy = x ** 2 + y ** 2

    r = f.sqrt(xy + z ** 2)
    nudge = (f.abs(z) < 1e-6) * 1e-6  # assure numerical stability
    t = f.arctan2(f.sqrt(xy), z + nudge)
    nudge = (f.abs(x) < 1e-6) * 1e-6  # assure numerical stability
    p = f.arctan2(y, x + nudge)

    return f.stack([r, t, p], -1)


def img_to_los_trv_azi(b, f=np):
    b_x, b_y, b_z = b[..., 0], b[..., 1], b[..., 2]

    B_los = b_z
    B_trv = (b_x ** 2 + b_y ** 2) ** 0.5
    azi = f.arctan2(b_x, b_y)

    b = f.stack([B_los, B_trv, azi], -1)
    return b


def los_trv_azi_to_img(b, ambiguous=False, f=np):
    B_los, B_trv, azi = b[..., 0], b[..., 1], b[..., 2]
    B_x = B_trv * f.sin(azi % f.pi) if ambiguous else B_trv * f.sin(azi)
    B_y = B_trv * f.cos(azi % f.pi) if ambiguous else B_trv * f.cos(azi)
    B_z = B_los
    b = f.stack([B_x, B_y, B_z], -1)
    return b
