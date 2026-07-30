import inspect

import torch

import nf2.extrapolate as extrapolate


def test_single_run_accepts_optional_meta_path():
    assert "meta_path" in inspect.signature(extrapolate.run).parameters


def test_reset_checkpoint_progress_preserves_state_and_optimizer(tmp_path):
    source_path = tmp_path / "source.ckpt"
    output_path = tmp_path / "initial.ckpt"
    checkpoint = {
        "epoch": 12,
        "global_step": 345,
        "loops": {"fit_loop": "progress"},
        "callbacks": {"checkpoint": "state"},
        "state_dict": {"model.weight": torch.tensor([1.0])},
        "optimizer_states": [{"state": {}}],
    }
    torch.save(checkpoint, source_path)

    assert extrapolate._is_lightning_checkpoint(source_path)
    assert extrapolate._reset_checkpoint_progress(source_path, output_path) == output_path

    reset = torch.load(output_path, map_location="cpu", weights_only=False)
    assert reset["epoch"] == 0
    assert reset["global_step"] == 0
    assert "loops" not in reset
    assert "callbacks" not in reset
    assert reset["state_dict"]["model.weight"].item() == 1.0
    assert reset["optimizer_states"] == [{"state": {}}]


def test_existing_data_module_is_reused_by_default(tmp_path, monkeypatch):
    save_path = tmp_path / "data_module.pkl"
    torch.save({"state": "existing"}, save_path)

    def fail_if_rebuilt(**kwargs):
        raise AssertionError("Existing data module should be reused")

    monkeypatch.setattr(extrapolate, "CartesianDataModule", fail_if_rebuilt)

    extrapolate._initialize_data_module(
        {"type": "cartesian", "work_path": str(tmp_path)}, save_path)

    assert torch.load(save_path, weights_only=False) == {"state": "existing"}


def test_reload_rebuilds_existing_data_module(tmp_path, monkeypatch):
    save_path = tmp_path / "data_module.pkl"
    torch.save({"state": "existing"}, save_path)

    monkeypatch.setattr(
        extrapolate,
        "CartesianDataModule",
        lambda **kwargs: {"state": "rebuilt", "work_path": kwargs["work_path"]},
    )

    extrapolate._initialize_data_module(
        {"type": "cartesian", "work_path": str(tmp_path)}, save_path, reload=True)

    assert torch.load(save_path, weights_only=False) == {
        "state": "rebuilt",
        "work_path": str(tmp_path),
    }
