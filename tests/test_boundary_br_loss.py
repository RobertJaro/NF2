import torch

from nf2.train.loss import BoundaryBrLoss, loss_module_mapping


def test_boundary_br_loss_uses_radial_component_after_transform():
    loss = BoundaryBrLoss(name="boundary_br", ds_id="boundary")
    b = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    transform = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    b_true = torch.tensor(
        [
            [1.5, 100.0, 100.0],
            [5.0, 100.0, 100.0],
        ]
    )
    b_err = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    br_loss = loss(b=b, b_true=b_true, transform=transform, b_err=b_err)

    torch.testing.assert_close(br_loss, torch.tensor([0.0625, 0.25]))
    single_component_loss = loss(b=torch.tensor([[2.0, 3.0, 4.0]]), b_true=torch.tensor([[1.5]]))
    torch.testing.assert_close(single_component_loss, torch.tensor([0.25]))
    assert loss_module_mapping["boundary_br"] is BoundaryBrLoss
