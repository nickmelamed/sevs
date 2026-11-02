import numpy as np
import pytest

from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.models.detector import Detection

GT = [("a", np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]


def det(box, conf, cls=1):
    return Detection(np.array(box, dtype=float), cls, conf)


def test_detection_on_a_crowd_region_is_ignored_not_a_false_positive():
    preds = [("a", [det([0, 0, 10, 10], 0.9), det([50, 50, 60, 60], 0.95)])]
    without = evaluate_detection_predictions(preds, GT, iou_thresholds=[0.5])
    crowd = {"a": (np.array([[40, 40, 80, 80]], dtype=float), np.array([1]))}
    with_crowd = evaluate_detection_predictions(preds, GT, iou_thresholds=[0.5], crowd=crowd)
    assert without.map50 == pytest.approx(0.5)
    assert with_crowd.map50 == 1.0


def test_crowd_region_of_another_class_does_not_ignore():
    preds = [("a", [det([0, 0, 10, 10], 0.9), det([50, 50, 60, 60], 0.95)])]
    crowd = {"a": (np.array([[40, 40, 80, 80]], dtype=float), np.array([2]))}
    assert evaluate_detection_predictions(preds, GT, iou_thresholds=[0.5], crowd=crowd).map50 == pytest.approx(0.5)


def test_only_the_top_detections_per_class_and_image_count():
    preds = [("a", [det([0, 0, 10, 10], 0.9), det([50, 50, 60, 60], 0.95)])]
    assert evaluate_detection_predictions(preds, GT, iou_thresholds=[0.5], max_dets=1).map50 == 0.0


def test_recall_is_interpolated_at_101_points():
    gts = [("a", np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float), np.array([1, 1, 1]))]
    preds = [("a", [det([0, 0, 10, 10], 0.9)])]
    # recall reaches 1/3, so 34 of 101 recall points (0, .01, ..., .33) have precision 1
    assert evaluate_detection_predictions(preds, gts, iou_thresholds=[0.5]).map50 == pytest.approx(34 / 101)


def test_default_thresholds_are_the_ten_coco_values():
    out = evaluate_detection_predictions([("a", [det([0, 0, 10, 8], 0.9)])], GT)
    assert out.map50 == 1.0
    assert out.map50_95 == pytest.approx(0.7)
