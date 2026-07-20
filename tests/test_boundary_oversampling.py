import numpy as np

from nf2.loader.base import MapDataset
from nf2.loader.oversampling import apply_boundary_oversampling


def test_boundary_oversampling_keeps_full_dataset_and_appends_weighted_samples(monkeypatch):
    tensors = {'coords': np.arange(12, dtype=np.float32).reshape(4, 3)}
    sampling_field = np.array([0, 1, 2, 3], dtype=np.float32)

    def choice(values, size, replace, p):
        assert replace is True
        assert size == 8
        assert np.isclose(p.sum(), 1)
        return np.full(size, 3)

    monkeypatch.setattr(np.random, 'choice', choice)

    oversampled = apply_boundary_oversampling(
        tensors,
        sampling_field,
        {'type': 'field_strength', 'factor': 2, 'floor': 0},
    )

    assert oversampled['coords'].shape == (12, 3)
    np.testing.assert_array_equal(oversampled['coords'][:4], tensors['coords'])
    np.testing.assert_array_equal(oversampled['coords'][4:], np.repeat(tensors['coords'][3:4], 8, axis=0))

    unchanged = apply_boundary_oversampling(
        {'coords': tensors['coords'].reshape(2, 2, 3)},
        np.ones((2, 2), dtype=np.float32),
        {'type': 'field_strength', 'factor': 0},
    )
    np.testing.assert_array_equal(unchanged['coords'], tensors['coords'])


def test_boundary_oversampling_smooths_sampling_field_before_weighting(monkeypatch):
    tensors = {'coords': np.arange(27, dtype=np.float32).reshape(3, 3, 3)}
    sampling_field = np.zeros((3, 3), dtype=np.float32)
    sampling_field[1, 1] = 1
    captured = {}

    def choice(values, size, replace, p):
        captured['p'] = p
        return np.zeros(size, dtype=np.int64)

    monkeypatch.setattr(np.random, 'choice', choice)

    apply_boundary_oversampling(
        tensors,
        sampling_field,
        {'type': 'field_strength', 'factor': 1, 'floor': 0, 'smooth_sigma': 1},
    )

    probabilities = captured['p'].reshape(3, 3)
    assert probabilities[1, 1] < 1
    assert probabilities[0, 1] > 0


def test_map_dataset_applies_field_strength_oversampling(tmp_path):
    b = np.zeros((4, 4, 3), dtype=np.float32)
    b[..., 2] = np.arange(16, dtype=np.float32).reshape(4, 4)

    dataset = MapDataset(
        b=b,
        batch_size=64,
        work_path=tmp_path,
        plot=False,
        shuffle=False,
        oversampling={'type': 'field_strength', 'factor': 1, 'field': 'normal', 'floor': 0},
    )

    batch = dataset[0]
    assert batch['coords'].shape == (32, 3)
    assert batch['b_true'].shape == (32, 3)
