import numpy as np


def spherical_to_cartesian_matrix(c):
    p, t, r = c[..., 0], c[..., 1], c[..., 2]
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
    p, t, r = c[..., 0], c[..., 1], c[..., 2]
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

def vector_spherical_to_cartesian(v, c):
    vp, vt, vr = v[..., 0], v[..., 1], v[..., 2]
    p, t, r = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    vx = vr * sin(t) * cos(p) + vt * cos(t) * cos(p) - vp * sin(p)
    vy = vr * sin(t) * sin(p) + vt * cos(t) * sin(p) + vp * cos(p)
    vz = vr * cos(t) - vt * sin(t)
    #
    return np.stack([vx, vy, vz], -1)

def vector_cartesian_to_spherical(v, c):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    p, t, r = c[..., 0], c[..., 1], c[..., 2]
    sin = np.sin
    cos = np.cos
    #
    vr = vx * sin(t) * cos(p) + vy * sin(t) * sin(p) + vz * cos(t)
    vt = vx * cos(t) * cos(p) + vy * cos(t) * sin(p) - vz * sin(t)
    vp = - vx * sin(p) + vy * cos(p)
    #
    return np.stack([vp, vt, vr], -1)

def spherical_to_cartesian(v):
    sin = np.sin
    cos = np.cos
    p, t, r = v[..., 0], v[..., 1], v[..., 2]
    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)
    return np.stack([x, y, z], -1)

def cartesian_to_spherical(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    xy = x ** 2 + y ** 2

    r = np.sqrt(xy + z**2)
    t = np.arctan2(np.sqrt(xy), z)
    p = np.arctan2(y, x)

    return np.stack([p, t, r], -1)