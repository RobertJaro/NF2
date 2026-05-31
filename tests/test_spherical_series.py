import pytest

from nf2.loader.spherical import SphericalSeriesDataModule
from nf2.loader.spherical import _series_work_path as spherical_series_work_path
from nf2.data.dataset import IndexedDataset, TensorsDataset
from nf2.train.callback import _output_cube_shape, _valid_output_cube_shape


def test_spherical_series_resolves_validation_placeholders_per_step():
    data_module = SphericalSeriesDataModule.__new__(SphericalSeriesDataModule)
    data_module.iterations = None
    full_disk_1 = "hmi.b_720s.20240322_000000_TAI.Br.fits"
    full_disk_2 = "hmi.b_720s.20240322_030000_TAI.Br.fits"

    boundaries = data_module._expand_boundaries(
        [
            {
                "id": "full_disk",
                "type": "map",
                "files": {
                    "Br": [full_disk_1, full_disk_2],
                    "Bt": [
                        "hmi.b_720s.20240322_000000_TAI.Bt.fits",
                        "hmi.b_720s.20240322_030000_TAI.Bt.fits",
                    ],
                    "Bp": [
                        "hmi.b_720s.20240322_000000_TAI.Bp.fits",
                        "hmi.b_720s.20240322_030000_TAI.Bp.fits",
                    ],
                    "Br_err": [
                        "hmi.b_720s.20240322_000000_TAI.Br_err.fits",
                        "hmi.b_720s.20240322_030000_TAI.Br_err.fits",
                    ],
                },
            },
            {
                "id": "synoptic",
                "type": "map",
                "files": {"Br": "syn_br.fits", "Bt": "syn_bt.fits", "Bp": "syn_bp.fits"},
                "mask_configs": {"type": "reference", "file": "[[full_disk.files.Br]]"},
            },
        ]
    )
    validation = data_module._expand_validation(
        [
            {
                "id": "full_disk_valid",
                "type": "map",
                "files": {
                    "Br": "[[full_disk.files.Br]]",
                    "Bt": "[[full_disk.files.Bt]]",
                    "Bp": "[[full_disk.files.Bp]]",
                    "Br_err": "[[full_disk.errors.Br_err]]",
                },
            },
            {
                "id": "synoptic_valid",
                "type": "map",
                "files": {
                    "Br": "[[synoptic.files.Br]]",
                    "Bt": "[[synoptic.files.Bt]]",
                    "Bp": "[[synoptic.files.Bp]]",
                },
            },
        ],
        boundaries,
    )

    assert boundaries[0][1]["mask_configs"]["file"] == full_disk_1
    assert boundaries[1][1]["mask_configs"]["file"] == full_disk_2
    assert validation[0][0]["files"]["Br"] == full_disk_1
    assert validation[1][0]["files"]["Br"] == full_disk_2
    assert validation[1][0]["files"]["Br_err"] == "hmi.b_720s.20240322_030000_TAI.Br_err.fits"
    assert validation[0][1]["files"]["Br"] == "syn_br.fits"


def test_indexed_dataset_attaches_rectangular_cube_shape():
    class Dataset:
        cube_shape = (12, 34)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"coords": idx}

    batch = IndexedDataset(Dataset())[0]

    assert batch["cube_shape"].tolist() == [[12, 34]]


def test_output_cube_shape_uses_validation_metadata_for_rectangular_map():
    outputs = {"cube_shape": [[12, 34]]}

    assert _output_cube_shape(outputs) == (12, 34)


def test_output_cube_shape_uses_flat_validation_metadata():
    outputs = {"cube_shape": [3741, 3742]}

    assert _output_cube_shape(outputs) == (3741, 3742)


def test_valid_output_cube_shape_rejects_stale_shape():
    class Tensor:
        shape = (3741 * 3742, 3)

    with pytest.warns(RuntimeWarning, match="Skipping boundary plot"):
        assert not _valid_output_cube_shape(Tensor(), (3743, 3743))


def test_spherical_series_step_kwargs_use_isolated_work_paths():
    data_module = SphericalSeriesDataModule.__new__(SphericalSeriesDataModule)
    data_module.kwargs = {"work_path": "/tmp/nf2-work", "validation": []}
    data_module.boundaries = [[{"id": "step_1"}], [{"id": "step_2"}]]
    data_module.validation = [[{"id": "valid_1"}], [{"id": "valid_2"}]]
    data_module.preload_data_modules = True

    assert data_module._kwargs_for_step(0)["work_path"] == "/tmp/nf2-work/series_data_modules/000000"
    assert data_module._kwargs_for_step(1)["work_path"] == "/tmp/nf2-work/series_data_modules/000001"
    assert data_module._kwargs_for_step(0)["overview_id"] == "step_1"
    assert data_module._kwargs_for_step(1)["overview_id"] == "step_2"


def test_spherical_series_work_path_adds_rank_only_when_requested(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")

    assert spherical_series_work_path("/tmp/nf2-work", 0) == "/tmp/nf2-work/series_data_modules/000000"
    assert spherical_series_work_path(
        "/tmp/nf2-work", 0, include_rank=True
    ) == "/tmp/nf2-work/series_data_modules/000000/rank_001"


def test_tensors_dataset_clear_ignores_missing_cache_file():
    dataset = TensorsDataset.__new__(TensorsDataset)
    dataset.file_paths = {"missing": "/tmp/nf2_missing_cache_file.npy"}

    dataset.clear()


def test_spherical_series_preloads_remaining_steps(monkeypatch):
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

    monkeypatch.setattr("nf2.loader.spherical._load_spherical_data_module", fake_load)
    monkeypatch.setattr("nf2.loader.spherical.ThreadPoolExecutor", FakeExecutor)

    data_module = SphericalSeriesDataModule.__new__(SphericalSeriesDataModule)
    data_module.args = ()
    data_module.kwargs = {"work_path": "/tmp/nf2-work"}
    data_module.preload_data_modules = True
    data_module.step = 0
    data_module.total_steps = 3
    data_module.boundaries = [[{"id": "step_0"}], [{"id": "step_1"}], [{"id": "step_2"}]]
    data_module.validation = [[{"id": "valid_0"}], [{"id": "valid_1"}], [{"id": "valid_2"}]]
    data_module.data_modules = [None, None, None]

    data_module._load_data_modules(data_module_workers=2)

    assert data_module.data_modules == ["module-0", "module-1", "module-2"]
    assert executor_steps == [0, 1, 2]
    assert sorted(calls) == [0, 1, 2]
