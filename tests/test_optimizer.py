import torch

from nf2.train.module import NF2Module
from nf2.train.soap import SOAP


def _module(optimizer):
    return NF2Module(
        validation_mapping={},
        data_config={"Mm_per_ds": 1, "Gauss_per_dB": 1},
        model_kwargs={"type": "b", "dim": 8, "n_layers": 1},
        loss_config=[],
        lr_params=optimizer,
    )


def test_soap_optimizer_can_be_selected_with_zero_weight_decay():
    module = _module({
        "type": "soap",
        "start": 1e-3,
        "end": 1e-4,
        "iterations": 100,
        "weight_decay": 0.5,
        "precondition_frequency": 5,
    })

    optimizers, _ = module.configure_optimizers()
    optimizer = optimizers[0]

    assert isinstance(optimizer, SOAP)
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["precondition_frequency"] == 5


def test_soap_optimizer_uses_default_learning_rate_schedule():
    module = _module({"type": "soap"})

    optimizers, schedulers = module.configure_optimizers()

    assert optimizers[0].param_groups[0]["lr"] == 5e-4
    assert module.lr_params == {
        "type": "soap", "start": 5e-4, "end": 5e-5, "iterations": 1e5}
    assert schedulers[0].gamma == (5e-5 / 5e-4) ** (1 / 1e5)


def test_adam_optimizer_remains_default_with_zero_weight_decay():
    module = _module({"start": 1e-3, "end": 1e-4, "iterations": 100})

    optimizers, _ = module.configure_optimizers()
    optimizer = optimizers[0]

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["weight_decay"] == 0


def test_unknown_optimizer_is_rejected():
    module = _module({"type": "unknown", "start": 1e-3, "end": 1e-4, "iterations": 100})

    try:
        module.configure_optimizers()
    except ValueError as exc:
        assert "Invalid optimizer type" in str(exc)
    else:
        raise AssertionError("Unknown optimizer type was accepted")
