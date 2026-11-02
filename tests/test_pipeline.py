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
