import numpy as np

from nf2.data.util import (
    cartesian_to_spherical,
    latitude_to_colatitude,
    spherical_to_cartesian,
    vector_cartesian_to_spherical,
    vector_spherical_to_cartesian,
)


def test_latitude_colatitude_roundtrip():
    latitude = np.array([-np.pi / 2, -0.3, 0.0, 0.4, np.pi / 2])
    colatitude = latitude_to_colatitude(latitude)
    recovered = latitude_to_colatitude(colatitude)
    assert np.allclose(latitude, recovered)


def test_spherical_cartesian_uses_colatitude_convention():
    spherical = np.array([1.2, latitude_to_colatitude(0.35), 1.1])
    cartesian = spherical_to_cartesian(spherical)
    recovered = cartesian_to_spherical(cartesian)
    assert np.allclose(spherical, recovered)


def test_vector_roundtrip_uses_same_spherical_convention():
    coords = np.array([1.4, latitude_to_colatitude(-0.2), 2.3])
    vector_rtp = np.array([3.0, -4.0, 5.0])
    vector_xyz = vector_spherical_to_cartesian(vector_rtp, coords)
    recovered = vector_cartesian_to_spherical(vector_xyz, coords)
    assert np.allclose(vector_rtp, recovered)
