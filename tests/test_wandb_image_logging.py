from matplotlib import pyplot as plt

from nf2.train.callback import _log_wandb_figure


def test_log_wandb_figure_wraps_matplotlib_figures(monkeypatch):
    calls = []

    def fake_image(fig):
        return ("image", fig)

    def fake_log(payload):
        calls.append(payload)

    monkeypatch.setattr("nf2.train.callback.wandb.Image", fake_image)
    monkeypatch.setattr("nf2.train.callback.wandb.log", fake_log)

    fig = plt.figure()
    try:
        _log_wandb_figure("example", fig)
    finally:
        plt.close(fig)

    assert calls == [{"example": ("image", fig)}]
