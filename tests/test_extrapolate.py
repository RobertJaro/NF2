import inspect

import torch

from nf2.extrapolate import _is_lightning_checkpoint, _reset_checkpoint_progress, run


def test_single_run_accepts_optional_meta_path():
    assert "meta_path" in inspect.signature(run).parameters


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

    assert _is_lightning_checkpoint(source_path)
    assert _reset_checkpoint_progress(source_path, output_path) == output_path

    reset = torch.load(output_path, map_location="cpu", weights_only=False)
    assert reset["epoch"] == 0
    assert reset["global_step"] == 0
    assert "loops" not in reset
    assert "callbacks" not in reset
    assert reset["state_dict"]["model.weight"].item() == 1.0
    assert reset["optimizer_states"] == [{"state": {}}]
