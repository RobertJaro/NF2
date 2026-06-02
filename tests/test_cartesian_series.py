from nf2.loader.cartesian import CartesianSeriesDataModule
from nf2.loader.cartesian import _series_work_path as cartesian_series_work_path


def test_cartesian_series_step_kwargs_use_isolated_work_paths():
    data_module = CartesianSeriesDataModule.__new__(CartesianSeriesDataModule)
    data_module.kwargs = {"work_path": "/tmp/nf2-work"}
    data_module.preload_data_modules = True

    assert data_module._kwargs_for_step(0)["work_path"] == "/tmp/nf2-work/series_data_modules/000000"
    assert data_module._kwargs_for_step(1)["work_path"] == "/tmp/nf2-work/series_data_modules/000001"


def test_cartesian_series_work_path_adds_rank_only_when_requested(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert cartesian_series_work_path("/tmp/nf2-work", 0) == "/tmp/nf2-work/series_data_modules/000000"
    assert cartesian_series_work_path(
        "/tmp/nf2-work", 0, include_rank=True
    ) == "/tmp/nf2-work/series_data_modules/000000/rank_001"


def test_cartesian_series_preloads_remaining_steps(monkeypatch):
    calls = []
    executor_steps = []

    def fake_load(args):
        step = args[0]
        calls.append(step)
        return f"module-{step}"

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def map(self, func, worker_args):
            worker_args = list(worker_args)
            executor_steps.extend(args[0] for args in worker_args)
            return [func(args) for args in worker_args]

    monkeypatch.setattr("nf2.loader.cartesian._load_cartesian_data_module", fake_load)
    monkeypatch.setattr("nf2.loader.cartesian.ThreadPoolExecutor", FakeExecutor)

    data_module = CartesianSeriesDataModule.__new__(CartesianSeriesDataModule)
    data_module.args = ()
    data_module.kwargs = {"work_path": "/tmp/nf2-work"}
    data_module.preload_data_modules = True
    data_module.step = 0
    data_module.total_steps = 3
    data_module.boundaries = [[{"id": "step_0"}], [{"id": "step_1"}], [{"id": "step_2"}]]
    data_module.data_modules = [None, None, None]

    data_module._load_data_modules(data_module_workers=2)

    assert data_module.data_modules == ["module-0", "module-1", "module-2"]
    assert executor_steps == [0, 1, 2]
    assert sorted(calls) == [0, 1, 2]


def test_cartesian_series_activate_step_rebuilds_lazy_module(monkeypatch):
    calls = []

    def fake_load(args):
        step = args[0]
        calls.append(step)
        return f"module-{step}"

    monkeypatch.setattr("nf2.loader.cartesian._load_cartesian_data_module", fake_load)

    data_module = CartesianSeriesDataModule.__new__(CartesianSeriesDataModule)
    data_module.args = ()
    data_module.kwargs = {"work_path": "/tmp/nf2-work"}
    data_module.preload_data_modules = False
    data_module.step = 0
    data_module.total_steps = 2
    data_module.boundaries = [[{"id": "step_0"}], [{"id": "step_1"}]]
    data_module.data_modules = ["stale-module", None]

    data_module.activate_step(0)

    assert data_module.data_modules == ["module-0", None]
    assert calls == [0]
