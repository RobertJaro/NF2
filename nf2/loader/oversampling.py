import numpy as np
from scipy.ndimage import gaussian_filter


def boundary_sampling_field(b, mode='magnitude'):
    mode = mode or 'magnitude'
    if mode == 'magnitude':
        valid = np.isfinite(b).any(axis=-1)
        field = np.sqrt(np.nansum(b ** 2, axis=-1))
        field[~valid] = np.nan
        return field
    if mode in ['normal', 'radial', 'z', 'Br', 'Bz']:
        return np.abs(b[..., 0] if mode in ['radial', 'Br'] else b[..., 2])
    raise ValueError(
        f"Unknown oversampling field '{mode}'. "
        "Supported fields are 'magnitude', 'normal', 'radial', 'z', 'Br', and 'Bz'."
    )


def apply_boundary_oversampling(tensors, sampling_field, oversampling):
    if oversampling is None:
        return tensors
    if not isinstance(oversampling, dict):
        raise TypeError('oversampling must be a configuration dictionary.')

    o_type = oversampling.get('type', 'field_strength')
    if o_type != 'field_strength':
        raise ValueError(f"Unknown oversampling type '{o_type}'. Supported types: ['field_strength'].")

    factor = float(oversampling.get('factor', 0))

    sampling_field = np.asarray(sampling_field, dtype=np.float64)
    sample_shape = sampling_field.shape
    n_samples = int(np.prod(sample_shape))
    if factor <= 0:
        return _flatten_tensors(tensors, sample_shape, n_samples)

    n_extra = int(round(factor * n_samples))
    if n_extra <= 0:
        return _flatten_tensors(tensors, sample_shape, n_samples)

    weights = _sampling_weights(
        sampling_field,
        exponent=float(oversampling.get('exponent', 1.0)),
        floor=float(oversampling.get('floor', 0.0)),
        smooth_sigma=float(oversampling.get('smooth_sigma', 0.0)),
    ).reshape(-1)

    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return _flatten_tensors(tensors, sample_shape, n_samples)
    probabilities = np.zeros_like(weights, dtype=np.float64)
    probabilities[valid] = weights[valid] / weights[valid].sum()

    extra_idx = np.random.choice(np.arange(n_samples), size=n_extra, replace=True, p=probabilities)
    full_idx = np.concatenate([np.arange(n_samples), extra_idx])

    flat_tensors = _flatten_tensors(tensors, sample_shape, n_samples)
    return {key: value[full_idx] for key, value in flat_tensors.items()}


def _flatten_tensors(tensors, sample_shape, n_samples):
    flattened = {}
    for key, value in tensors.items():
        value = np.asarray(value)
        if value.shape[:len(sample_shape)] != sample_shape:
            raise ValueError(
                f"Tensor '{key}' has leading shape {value.shape[:len(sample_shape)]}, "
                f"expected sampling shape {sample_shape}."
            )
        flattened[key] = value.reshape((n_samples, *value.shape[len(sample_shape):]))
    return flattened


def _sampling_weights(sampling_field, exponent=1.0, floor=0.05, smooth_sigma=0.0):
    if exponent < 0:
        raise ValueError('oversampling exponent must be non-negative.')
    if floor < 0:
        raise ValueError('oversampling floor must be non-negative.')
    if smooth_sigma < 0:
        raise ValueError('oversampling smooth_sigma must be non-negative.')

    field = np.asarray(sampling_field, dtype=np.float64)
    if smooth_sigma > 0:
        field = _nan_gaussian_filter(field, smooth_sigma)

    valid = np.isfinite(field)
    weights = np.full(field.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return weights

    clipped = np.clip(field[valid], 0, None)
    weights[valid] = clipped ** exponent
    mean_weight = np.nanmean(weights[valid])
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        weights[valid] = 1.0
    else:
        weights[valid] += floor * mean_weight
    return weights


def _nan_gaussian_filter(values, sigma):
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full(values.shape, np.nan, dtype=np.float64)

    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma)
    denominator = gaussian_filter(valid.astype(np.float64), sigma=sigma)
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    good = denominator > np.finfo(np.float64).eps
    smoothed[good] = numerator[good] / denominator[good]
    return smoothed
