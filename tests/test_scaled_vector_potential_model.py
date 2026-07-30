import torch

from nf2.train.model import ScaledVectorPotentialModel, SirenModel, curl


def test_scaled_vector_potential_uses_default_power_laws():
    torch.manual_seed(0)
    model = ScaledVectorPotentialModel(dim=8, n_layers=1, base_radius=2.0)
    coords = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )

    network_coords = coords * (coords.pow(2).sum(-1, keepdim=True).sqrt() / model.base_radius).pow(
        -model.coordinate_radial_power
    )
    raw_a = SirenModel.forward(model, network_coords)
    out = model(coords, compute_jacobian=False)

    assert model.radial_power == 2.0
    assert model.coordinate_radial_power == 4.0
    torch.testing.assert_close(out["network_coords"], network_coords)
    torch.testing.assert_close(out["a"][0], raw_a[0])
    torch.testing.assert_close(out["a"][1], raw_a[1] * 0.25)

    unscaled_coords_model = ScaledVectorPotentialModel(
        dim=8, n_layers=1, base_radius=2.0, coordinate_radial_power=0.0
    )
    unscaled_out = unscaled_coords_model(coords[1:2].detach().requires_grad_(), compute_jacobian=False)
    torch.testing.assert_close(unscaled_out["network_coords"], coords[1:2])


def test_scaled_vector_potential_curls_scaled_a():
    torch.manual_seed(0)
    model = ScaledVectorPotentialModel(dim=8, n_layers=1, base_radius=2.0)
    coords = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.5],
        ],
        requires_grad=True,
    )

    radius = coords.pow(2).sum(-1, keepdim=True).sqrt()
    normalized_radius = radius / model.base_radius
    network_coords = coords * normalized_radius.pow(-model.coordinate_radial_power)
    raw_a = SirenModel.forward(model, network_coords)
    scaled_a = raw_a * normalized_radius.pow(-model.radial_power)
    expected_b = curl(scaled_a, coords)

    out = model(coords, compute_jacobian=False)

    torch.testing.assert_close(out["b"], expected_b)
