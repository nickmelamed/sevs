"""End-to-end runs of the pipeline on synthetic data."""
import copy

import numpy as np
import pytest

from sevs.cli import run_pipeline
from sevs.models.detector import BaseDetector, Detection
from sevs.utils.io import load_yaml


class JitterDetector(BaseDetector):
    """Return the true box with noise, plus one wrong-class box, per image."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, image):
        h, w = image.shape[:2]
        dets = []
        for _ in range(3):
            x1, y1 = self.rng.uniform(0, w * 0.5), self.rng.uniform(0, h * 0.5)
            box = np.array([x1, y1, x1 + w * 0.3, y1 + h * 0.3])
            noise = self.rng.normal(0, 8, 4)
            dets.append(Detection(box + noise, 1, float(self.rng.uniform(0.3, 0.99))))
        dets.append(Detection(np.array([0.0, 0.0, 5.0, 5.0]), 2, float(self.rng.uniform(0.05, 0.4))))
        return dets


def make_samples(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        x1, y1 = rng.uniform(0, 40), rng.uniform(0, 30)
        box = np.array([[x1, y1, x1 + 48, y1 + 36]])
        out.append(
            {"image": image, "image_id": f"img{i}", "gt_boxes_xyxy": box, "gt_labels": np.array([1]), "meta": {}}
        )
    return out


@pytest.fixture
def cfg(tmp_path):
    c = load_yaml("configs/eval/eval_default.yaml")
    c["run"]["output_dir"] = str(tmp_path / "out")
    return c


def test_mock_smoke_row_is_pinned(cfg):
    row = run_pipeline(cfg)
    assert row["n_images"] == 12 and row["n_detections"] == 38
    assert row["ece_conf"] == pytest.approx(0.5226836528661616)
    assert row["brier_conf"] == pytest.approx(0.34746134366115977)
    assert np.isnan(row["meta_error_auroc"])


def test_run_is_deterministic(cfg):
    a = run_pipeline(copy.deepcopy(cfg), samples=make_samples(), detector=JitterDetector(1))
    b = run_pipeline(copy.deepcopy(cfg), samples=make_samples(), detector=JitterDetector(1))
    assert a == b or all(
        (x == y) or (isinstance(x, float) and np.isnan(x) and np.isnan(y)) for x, y in ((a[k], b[k]) for k in a)
    )


def test_synthetic_run_produces_a_meta_auroc(cfg):
    cfg["detector"]["nms_iou_threshold"] = None
    row = run_pipeline(cfg, samples=make_samples(), detector=JitterDetector(1))
    assert row["n_images"] == 40 and row["n_detections"] == 160
    assert 0.0 <= row["meta_error_auroc"] <= 1.0
    assert 0.0 <= row["meta_error_auprc"] <= 1.0
    assert row["n_train_images"] + row["n_val_images"] + row["n_test_images"] == 40
    assert row["n_train_images"] == 28
    assert 0.0 <= row["abstain_coverage_0.5"] <= 1.0


class BrightBoxDetector(BaseDetector):
    """Detect the bounding box of the bright pixels, so it follows any geometric change."""

    def predict(self, image):
        ys, xs = np.where(image[..., 0] > 200)
        if len(xs) == 0:
            return []
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
        return [Detection(box, 1, 0.9)]


def test_tta_detections_come_back_in_the_original_frame():
    from sevs.cli import predict_tta

    image = np.zeros((100, 160, 3), dtype=np.uint8)
    image[20:50, 30:90] = 255
    method = {"augmentations": ["hflip", "scale_0.9", "scale_1.1", "brightness_1.2"], "num_samples": 5}
    preds = predict_tta(BrightBoxDetector(), image, method)
    assert len(preds) == 5
    for dets in preds:
        assert dets[0].box_xyxy == pytest.approx([30, 20, 90, 50], abs=3.0)


class DuplicateDetector(BaseDetector):
    """Report the true box twice, with confidence 0.9 and 0.8."""

    def predict(self, image):
        h, w = image.shape[:2]
        box = np.array([10.0, 10.0, 60.0, 50.0])
        return [Detection(box.copy(), 1, 0.9), Detection(box.copy(), 1, 0.8)]


def test_duplicate_detections_are_labeled_incorrect(cfg):
    cfg["detector"]["nms_iou_threshold"] = None
    samples = [
        {
            "image": np.zeros((100, 120, 3), dtype=np.uint8),
            "image_id": f"i{i}",
            "gt_boxes_xyxy": np.array([[10.0, 10.0, 60.0, 50.0]]),
            "gt_labels": np.array([1]),
            "meta": {},
        }
        for i in range(2)
    ]
    row = run_pipeline(cfg, samples=samples, detector=DuplicateDetector())
    # per image: labels [1, 0] against confidences [0.9, 0.8]
    assert row["brier_conf"] == pytest.approx(((0.1) ** 2 + 0.8**2) / 2)


def duplicate_samples():
    return [
        {
            "image": np.zeros((100, 120, 3), dtype=np.uint8),
            "image_id": f"i{i}",
            "gt_boxes_xyxy": np.array([[10.0, 10.0, 60.0, 50.0]]),
            "gt_labels": np.array([1]),
            "meta": {},
        }
        for i in range(2)
    ]


def test_nms_setting_removes_duplicates_before_labeling(cfg):
    row = run_pipeline(cfg, samples=duplicate_samples(), detector=DuplicateDetector())
    assert row["n_detections"] == 2


def test_max_detections_and_confidence_threshold_are_applied(cfg):
    cfg["detector"]["nms_iou_threshold"] = None
    cfg["detector"]["max_detections_per_image"] = 1
    assert run_pipeline(copy.deepcopy(cfg), samples=duplicate_samples(), detector=DuplicateDetector())["n_detections"] == 2
    cfg["detector"]["max_detections_per_image"] = None
    cfg["detector"]["conf_threshold"] = 0.85
    assert run_pipeline(cfg, samples=duplicate_samples(), detector=DuplicateDetector())["n_detections"] == 2


def test_class_filter_removes_other_classes(cfg):
    cfg["dataset"]["class_filter"] = [2]
    row = run_pipeline(cfg, samples=make_samples(5), detector=JitterDetector(1))
    assert row["n_detections"] == 5


def test_disabled_sections_leave_no_columns(cfg):
    cfg["evaluation"]["calibration"]["enabled"] = False
    cfg["evaluation"]["abstention"]["enabled"] = False
    cfg["meta_error_predictor"]["enabled"] = False
    row = run_pipeline(cfg, samples=make_samples(5), detector=JitterDetector(1))
    assert "ece_conf" not in row and "meta_error_auroc" not in row
    assert not any(k.startswith("abstain_") for k in row)


def test_listed_metrics_choose_columns(cfg):
    cfg["evaluation"]["calibration"]["metrics"] = ["ece"]
    cfg["evaluation"]["error_detection"]["metrics"] = ["auprc"]
    row = run_pipeline(cfg, samples=make_samples(20), detector=JitterDetector(1))
    assert "ece_conf" in row and "brier_conf" not in row
    assert "meta_error_auprc" in row and "meta_error_auroc" not in row


def test_detection_map_uses_the_configured_ground_truth(cfg):
    cfg["evaluation"]["detection"] = {"enabled": True, "metrics": ["mAP50"]}
    cfg["detector"]["nms_iou_threshold"] = None
    row = run_pipeline(cfg, samples=duplicate_samples(), detector=DuplicateDetector())
    assert row["map50"] == 1.0 and "map50_95" not in row


def test_outputs_list_controls_files(cfg, tmp_path):
    cfg["reporting"]["outputs"] = ["results_table.csv"]
    run_pipeline(cfg, samples=make_samples(20), detector=JitterDetector(1))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["results_table.csv"]
    cfg["reporting"]["outputs"] = ["results_table.csv", "reliability_diagram.png", "roc_error_detection.png"]
    run_pipeline(cfg, samples=make_samples(20), detector=JitterDetector(1))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "reliability_diagram.png",
        "results_table.csv",
        "roc_error_detection.png",
    ]


def test_reporting_disabled_writes_nothing(cfg, tmp_path):
    cfg["reporting"]["enabled"] = False
    run_pipeline(cfg, samples=make_samples(5), detector=JitterDetector(1))
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
