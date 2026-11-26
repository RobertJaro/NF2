"""Finite-volume relative magnetic helicity on a right-handed Cartesian grid.

Uses the Finn–Antonsen integral and the finite-box DeVore gauge of Valori,
Démoulin & Pariat (2012), doi:10.1007/s11207-012-9951-6.
See also Valori et al. (2016), doi:10.1007/s11214-016-0299-3.
"""
import argparse
import json

import numpy as np
from scipy.fft import dctn, idctn
from scipy.integrate import cumulative_trapezoid


def _weights(n, step):
    weights = np.full(n, step, dtype=float)
    weights[[0, -1]] *= 0.5
    return weights


def _face_fluxes(weights, normal_component):
    """Integrate on nodal faces; normal_component returns the Cartesian component."""
    net_flux = unsigned_flux = area = 0.
    for axis in range(3):
        face_weights = [w for i, w in enumerate(weights) if i != axis]
        area += 2 * face_weights[0].sum() * face_weights[1].sum()
        for index, sign in ((0, -1), (-1, 1)):
            normal = normal_component(axis, index)
            if not np.isfinite(normal).all():
                raise ValueError("Non-finite field on the boundary of the evaluation box.")
            net_flux += sign * np.einsum('i,j,ij->', *face_weights, normal)
            unsigned = np.einsum('i,j,ij->', *face_weights, abs(normal))
            unsigned_flux += unsigned
            if axis == 2 and index == 0:
                bottom_unsigned = unsigned
    return {
        'net_flux_Mx': net_flux * 1e16,
        'unsigned_flux_Mx': unsigned_flux * 1e16,
        'bottom_unsigned_flux_Mx': bottom_unsigned * 1e16,
        'flux_imbalance_fraction': abs(net_flux) / unsigned_flux if unsigned_flux else 0.,
        'outward_field_offset_removed_G': net_flux / area,
    }


def _potential_reference(b, spacing, weights):
    """Solve Δφ=0 with ∂φ/∂n=B·n on all faces, using ghost nodes and DCT-I."""
    fluxes = _face_fluxes(weights, lambda axis, index: np.take(b[..., axis], index, axis=axis))
    # Subtract a uniform normal offset to balance the Neumann boundary data.
    offset = fluxes['outward_field_offset_removed_G']
    shape = b.shape[:3]
    rhs = np.zeros(shape)
    eigenvalues = []
    for axis, (count, step) in enumerate(zip(shape, spacing)):
        for index, sign in ((0, -1), (-1, 1)):
            face = [slice(None)] * 3
            face[axis] = index
            normal = sign * b[..., axis][tuple(face)]
            rhs[tuple(face)] -= 2 * (normal - offset) / step
        eigenvalues.append(-4 * (np.sin(np.arange(count) * np.pi / (2 * (count - 1))) / step) ** 2)
    spectrum = dctn(rhs, type=1, overwrite_x=True)
    laplacian = eigenvalues[0][:, None, None] + eigenvalues[1][None, :, None] + eigenvalues[2]
    laplacian[0, 0, 0] = 1
    spectrum[0, 0, 0] = 0  # fix the arbitrary additive constant in φ
    spectrum /= laplacian
    del laplacian
    phi = idctn(spectrum, type=1, overwrite_x=True)
    bp = np.empty_like(b)
    for axis, step in enumerate(spacing):
        bp[..., axis] = np.gradient(phi, step, axis=axis, edge_order=2)
        component = np.moveaxis(bp[..., axis], axis, 0)
        original = np.moveaxis(b[..., axis], axis, 0)
        component[0] = original[0] + offset
        component[-1] = original[-1] - offset
    return bp, fluxes


def _devore_integrals(b, bp, spacing, weights, from_top):
    dx, dy, dz = spacing
    indices = range(b.shape[2] - 1, -1, -1) if from_top else range(b.shape[2])
    # DeVore gauge Az=0. On the integration face choose ∂x Ay - ∂y Ax = Bz.
    def face_potential(field):
        bz = field[:, :, indices[0], 2]
        return np.stack((-0.5 * cumulative_trapezoid(bz, dx=dy, axis=1, initial=0),
                         0.5 * cumulative_trapezoid(bz, dx=dx, axis=0, initial=0)), axis=-1)

    a, ap = face_potential(b), face_potential(bp)
    hj = hpj = 0.
    previous = indices[0]
    for k in indices:
        if k != previous:
            half_step = (k - previous) * dz / 2
            for field, potential in ((b, a), (bp, ap)):
                potential[..., 0] += half_step * (field[:, :, k, 1] + field[:, :, previous, 1])
                potential[..., 1] -= half_step * (field[:, :, k, 0] + field[:, :, previous, 0])
        bj_xy = b[:, :, k, :2] - bp[:, :, k, :2]
        hj += weights[2][k] * np.einsum('i,j,ijc,ijc->', *weights[:2], a - ap, bj_xy)
        hpj += 2 * weights[2][k] * np.einsum('i,j,ijc,ijc->', *weights[:2], ap, bj_xy)
        previous = k
    # G² Mm⁴ to Mx². H_V = ∫(A+Ap)·(B-Bp) dV = H_J + H_PJ.
    return {'H_V_Mx2': (hj + hpj) * 1e32, 'H_J_Mx2': hj * 1e32, 'H_PJ_Mx2': hpj * 1e32}


def relative_helicity(b_gauss, spacing_Mm, *, directions=("bottom", "top")):
    """Evaluate a nodal (nx, ny, nz, 3) B array in G, with uniform steps in Mm.

    The first and last nodes lie on each physical face. Scalar or per-axis
    spacings are accepted. Both integration directions estimate the same H_V.
    A flux imbalance requires modified reference boundary values and internal
    divergence can make the result gauge dependent. Inspect the diagnostics.
    """
    b = np.asarray(b_gauss, dtype=np.float64)
    spacing = np.broadcast_to(np.asarray(spacing_Mm, dtype=float), (3,))
    if b.ndim != 4 or b.shape[-1] != 3 or min(b.shape[:3]) < 3:
        raise ValueError('Expected (nx, ny, nz, 3) with at least three nodes per axis.')
    if not np.isfinite(b).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Field must be finite and spacings finite and positive.')
    weights = [_weights(n, step) for n, step in zip(b.shape[:3], spacing)]
    if any(face not in ("bottom", "top") for face in directions):
        raise ValueError("Unknown helicity integration direction.")
    if directions:
        bp, diagnostics = _potential_reference(b, spacing, weights)
    else:
        diagnostics = _face_fluxes(weights, lambda axis, index: np.take(b[..., axis], index, axis=axis))
    div = np.zeros(b.shape[:3])
    for axis, step in enumerate(spacing):
        div += np.gradient(b[..., axis], step, axis=axis, edge_order=2)
    volume = np.prod([w.sum() for w in weights])
    diagnostics['rms_div_B_G_per_Mm'] = np.sqrt(np.einsum('i,j,k,ijk,ijk->', *weights, div, div) / volume)
    result = {'grid_shape': list(b.shape[:3]), 'spacing_Mm': spacing.tolist(), 'diagnostics': diagnostics}
    for face in directions:
        result['from_' + face] = _devore_integrals(b, bp, spacing, weights, from_top=face == 'top')
    return result


def evaluate_box(out, spacing_Mm=None, batch_size=4096, *, x_range=None, y_range=None,
                 height_range=None, flux_only=False, directions=("bottom", "top"), progress=False):
    """Sample the same physical faces at every resolution, including a requested crop."""
    from tqdm import tqdm

    target = out.Mm_per_pixel if spacing_Mm is None else spacing_Mm
    if not np.isfinite(target) or target <= 0 or batch_size < 1:
        raise ValueError('Grid spacing and batch size must be positive.')
    xy = np.asarray(out.coord_range[:2]) * out.Mm_per_ds
    box = np.array([xy[0] if x_range is None else x_range,
                    xy[1] if y_range is None else y_range,
                    out.data_config.get('z_range', [0., out.max_height]) if height_range is None else height_range])
    extent = box[:, 1] - box[:, 0]
    if not np.all(np.isfinite(box)) or np.any(extent <= 0):
        raise ValueError('Expected finite increasing ranges for a three-dimensional box.')
    intervals = np.maximum(2, np.ceil(extent / target).astype(int))
    axes = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(box, intervals)]
    spacing = extent / intervals
    shape = tuple(len(axis) for axis in axes)

    def sample_plane(axis, index):
        nodes = axes.copy()
        nodes[axis] = nodes[axis][[index]]
        coords = np.stack(np.meshgrid(*nodes, indexing='ij'), axis=-1).squeeze(axis)
        plane = out.load_coords(coords / out.Mm_per_ds, compute_jacobian=False, batch_size=batch_size)
        return plane['b'].to_value('G')

    grid = {'box_Mm': box.tolist(), 'grid_shape': list(shape), 'spacing_Mm': spacing.tolist()}
    if flux_only:
        weights = [_weights(n, h) for n, h in zip(shape, spacing)]
        diagnostics = _face_fluxes(weights, lambda axis, index: sample_plane(axis, index)[..., axis])
        return {**grid, 'diagnostics': diagnostics}

    b = np.empty((*shape, 3), dtype=np.float64)
    for k in tqdm(range(shape[2]), desc="Sample helicity box", disable=not progress):
        b[:, :, k] = sample_plane(2, k)
    return {**grid, **relative_helicity(b, spacing, directions=directions)}


def evaluate_checkpoint(path, spacing_Mm=None, batch_size=4096):
    """Compute relative helicity for a Cartesian .nf2 checkpoint."""
    import nf2

    out = nf2.load(path)
    if out.state['data']['type'] != 'cartesian':
        raise ValueError('Finite-volume helicity requires a Cartesian checkpoint.')
    return evaluate_box(out, spacing_Mm, batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Cartesian .nf2 file')
    parser.add_argument('--spacing-Mm', type=float, help='Target grid spacing in Mm (default: native pixel size)')
    parser.add_argument('--batch-size', type=int, default=4096, help='Coordinates per network evaluation')
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.spacing_Mm, args.batch_size), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
