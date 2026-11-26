import numpy as np
import pytest

from nf2.evaluation.helicity import _potential_reference, _weights, relative_helicity


def test_harmonic_potential_on_six_faces():
    # φ = xyz + x² - y² is harmonic with nonconstant normal data on every face.
    shape = (11, 9, 13)
    spacing = (0.21, 0.35, 0.14)
    x, y, z = np.meshgrid(*(np.arange(n) * d for n, d in zip(shape, spacing)), indexing='ij')
    b = np.stack((y*z + 2*x, x*z - 2*y, x*y), axis=-1)
    weights = [_weights(n, d) for n, d in zip(shape, spacing)]
    bp, diag = _potential_reference(b, spacing, weights)
    np.testing.assert_allclose(bp, b, atol=3e-12)
    assert diag['flux_imbalance_fraction'] < 1e-14
    for direction in ('from_bottom', 'from_top'):
        assert abs(relative_helicity(b, spacing)[direction]['H_V_Mx2']) / 1e32 < 1e-11


@pytest.mark.parametrize('sign', [-1, 1])
def test_signed_mutual_helicity_converges(sign):
    # B = B0 ez + curl(ψ ez), ψ = sign sin(πx/Lx) sin(πy/Ly).
    # ψ vanishes on the lateral walls, so Bp=B0 ez, HJ=0, HV=2 B0 ∫ψ.
    lengths = np.array([1., 2., 3.])
    expected = sign * 8 * 2 * np.prod(lengths) / np.pi**2 * 1e32
    errors = []
    for n in (16, 32):
        x, y, z = np.meshgrid(*(np.linspace(0, length, n+1) for length in lengths), indexing='ij')
        bx = sign * np.pi/2 * np.sin(np.pi*x) * np.cos(np.pi*y/2)
        by = -sign * np.pi * np.cos(np.pi*x) * np.sin(np.pi*y/2)
        result = relative_helicity(np.stack((bx, by, np.full_like(z, 2)), axis=-1), lengths/n)
        for direction in ('from_bottom', 'from_top'):
            assert abs(result[direction]['H_J_Mx2'] / expected) < 1e-12
            assert result[direction]['H_V_Mx2'] / expected == pytest.approx(1., abs=0.008)
        errors.append(abs(result['from_top']['H_V_Mx2'] / expected - 1))
    assert errors[1] < 0.27 * errors[0]


def test_closed_helical_field():
    # A=f(cos(kz), sin(kz), 0), f=sin(πx)sin(πy)sin(πz).
    # B=curl A has zero normal flux on each face and A·B=-k f².
    n = 40
    x, y, z = np.meshgrid(*[np.linspace(0, 1, n+1)]*3, indexing='ij')
    k = 2*np.pi
    horizontal = np.sin(np.pi*x)*np.sin(np.pi*y)
    f = horizontal*np.sin(np.pi*z)
    fz = np.pi*horizontal*np.cos(np.pi*z)
    fx = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    fy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
    c, s = np.cos(k*z), np.sin(k*z)
    b = np.stack((-fz*s-k*f*c, fz*c-k*f*s, fx*s-fy*c), axis=-1)
    result = relative_helicity(b, 1/n)
    for direction in ('from_bottom', 'from_top'):
        assert result[direction]['H_J_Mx2'] / (-k/8*1e32) == pytest.approx(1., abs=0.005)
        assert abs(result[direction]['H_PJ_Mx2']) / 1e32 < 1e-12


def test_unbalanced_field_reports_boundary_correction():
    n = 10
    b = np.zeros((n+1, n+1, n+1, 3))
    b[..., 2] = np.linspace(0, 1, n+1)
    diagnostics = relative_helicity(b, 1/n)['diagnostics']
    assert diagnostics['flux_imbalance_fraction'] == pytest.approx(1.)
    assert diagnostics['outward_field_offset_removed_G'] == pytest.approx(1/6)
    assert diagnostics['rms_div_B_G_per_Mm'] == pytest.approx(1.)


def test_checkpoint_with_known_helicity(tmp_path):
    import torch
    from nf2.evaluation.helicity import evaluate_checkpoint
    from nf2.train.model import BModel

    # Encode ψ=sin(πx)sin(πy) exactly using two SIREN neurons.
    model = BModel(dim=2, n_layers=1, w0_init=1)
    with torch.no_grad():
        model.in_layer.weight.copy_(2*np.pi * torch.tensor([[1., 1., 0.], [1., -1., 0.]]))
        model.in_layer.bias.zero_()
        model.out_layer.weight.copy_(np.pi/20 * torch.tensor([[1., 1.], [-1., 1.], [0., 0.]]))
        model.out_layer.bias.copy_(torch.tensor([0., 0., 0.2]))
    checkpoint = tmp_path / 'known.nf2'
    torch.save({'model': model, 'data': {'type': 'cartesian', 'Gauss_per_dB': 10.,
               'Mm_per_ds': 2., 'ds_per_pixel': [0.025], 'coord_range': [np.array([[0., 0.5], [0., 0.5]])],
               'max_height': 2., 'z_range': [1., 2.]}}, checkpoint)
    result = evaluate_checkpoint(checkpoint)
    assert result['box_Mm'] == [[0., 1.], [0., 1.], [1., 2.]]
    assert result['grid_shape'] == [21, 21, 21]
    assert result['from_top']['H_V_Mx2'] / (16/np.pi**2*1e32) == pytest.approx(1., abs=0.005)


def test_vector_potential_checkpoint_matches_analytic_curl(tmp_path):
    import torch
    from nf2.evaluation.helicity import evaluate_checkpoint
    from nf2.train.model import VectorPotentialModel

    # A=(0, sin(x), sin(x)sin(y)). With z-independent A its curl is known.
    model = VectorPotentialModel(dim=3, n_layers=1, w0_init=1)
    with torch.no_grad():
        model.in_layer.weight.copy_(torch.tensor([[1., 0., 0.], [1., -1., 0.], [1., 1., 0.]]))
        model.in_layer.bias.copy_(torch.tensor([0., np.pi/2, np.pi/2]))
        model.out_layer.weight.copy_(torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0.5, -0.5]]))
        model.out_layer.bias.zero_()
    path = tmp_path / 'vector.nf2'
    torch.save({'model': model, 'data': {'type': 'cartesian', 'Gauss_per_dB': 10.,
               'Mm_per_ds': 1., 'ds_per_pixel': 0.05, 'coord_range': [[0., 1.], [0., 1.]],
               'max_height': 1.}}, path)
    nodes = np.linspace(0, 1, 21)
    x, y, z = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    field = np.stack((np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y), np.cos(x)), axis=-1)
    expected = relative_helicity(10 * field, 0.05)
    actual = evaluate_checkpoint(path, batch_size=77)
    for direction in ('from_bottom', 'from_top'):
        assert actual[direction]['H_V_Mx2'] == pytest.approx(expected[direction]['H_V_Mx2'], rel=2e-5)
