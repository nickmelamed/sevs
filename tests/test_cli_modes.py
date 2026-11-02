import copy

import pytest

from sevs.cli import is_mock, main, run_pipeline, smoke_config
from sevs.utils.io import load_yaml

REAL = "configs/eval/eval_real_torchvision_coco.yaml"
MOCK = "configs/eval/eval_default.yaml"


def test_is_mock_detects_either_side():
    cfg = load_yaml(MOCK)
    assert is_mock(cfg)
    real = load_yaml(REAL)
    assert not is_mock(real)
    half = copy.deepcopy(real)
    half["detector"]["family"] = "mock"
    assert is_mock(half)
    half = copy.deepcopy(real)
    half["dataset"]["name"] = "mock"
    assert is_mock(half)


def test_results_row_says_whether_the_run_was_mock(tmp_path):
    cfg = load_yaml(MOCK)
    cfg["run"]["output_dir"] = str(tmp_path)
    assert run_pipeline(cfg)["is_mock"] is True


def test_smoke_config_forces_mock_and_keeps_real_results_apart():
    cfg = smoke_config(load_yaml(REAL))
    assert cfg["dataset"]["name"] == "mock" and cfg["detector"]["family"] == "mock"
    assert cfg["run"]["name"].endswith("_smoke")
    assert cfg["run"]["output_dir"].endswith("_smoke")


def test_run_refuses_a_mock_config(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--config", MOCK])
    assert e.value.code != 0
    assert "mock" in str(e.value.code).lower() or "mock" in capsys.readouterr().err.lower()


def test_smoke_test_runs_from_a_real_config_without_touching_data():
    row = main(["smoke-test", "--config", REAL])
    assert row["is_mock"] is True and row["run_name"].endswith("_smoke")
