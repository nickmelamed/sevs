"""Cases found by mutation testing: boundary values, defaults, and crowd overlap."""
import numpy as np
import pytest

from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.models.detector import Detection
from sevs.utils.geometry import match_detection_to_ground_truth
from sevs.utils.metrics import expected_calibration_error
from sevs.utils.splits import split_image_ids

GT = np.array([[0, 0, 10, 10]], dtype=float)
LABELS = np.array([1])


def test_match_defaults_are_half_iou_and_class_required():
    hit = np.array([0, 0, 10, 6], dtype=float)
    assert match_detection_to_ground_truth(hit, 1, GT, LABELS)[0] is True
    assert match_detection_to_ground_truth(hit, 2, GT, LABELS)[0] is False
    miss = np.array([0, 0, 10, 4], dtype=float)
    assert match_detection_to_ground_truth(miss, 1, GT, LABELS)[0] is False


def test_match_at_exactly_the_threshold_counts():
    half = np.array([0, 0, 10, 5], dtype=float)
    assert match_detection_to_ground_truth(half, 1, GT, LABELS, 0.5, True)[0] is True


def test_no_candidate_is_never_a_match_even_at_threshold_zero():
    out = match_detection_to_ground_truth(GT[0], 1, np.zeros((0, 4)), np.zeros(0, dtype=int), 0.0, True)
    assert out == (False, 0.0, -1)
    out = match_detection_to_ground_truth(GT[0], 3, GT, LABELS, 0.0, True)
    assert out[0] is False and out[2] == -1


CLS_GT = [("a", GT, LABELS)]


def det(box, conf=0.9):
    return Detection(np.array(box, dtype=float), 1, conf)


def crowd_at(box):
    return {"a": (np.array([box], dtype=float), np.array([1]))}


def run(preds, crowd, thr=0.5):
    return evaluate_detection_predictions([("a", preds)], CLS_GT, iou_thresholds=[thr], crowd=crowd).map50_95


def test_detection_far_from_the_crowd_region_is_still_a_false_positive():
    assert run([det([100, 100, 110, 110], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 40, 80, 80])) == pytest.approx(0.5)


def test_detection_half_inside_the_crowd_region_is_ignored_at_ioa_half():
    assert run([det([30, 0, 50, 10], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 0, 80, 10])) == 1.0


def test_detection_mostly_outside_the_crowd_region_is_a_false_positive():
    assert run([det([30, 0, 50, 10], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([45, 0, 80, 10])) == pytest.approx(0.5)


def test_tiny_detection_inside_the_crowd_region_is_ignored():
    box = [50, 50, 50.5, 51]
    assert run([det(box, 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 40, 80, 80]), thr=0.75) == 1.0


def test_zero_area_detection_is_not_ignored_by_a_crowd_region():
    assert run([det([50, 50, 50, 50], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 40, 80, 80])) == pytest.approx(0.5)


def test_ece_depends_on_the_bin_edges():
    y = np.array([0, 1])
    p = np.array([0.05, 0.95])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.05)


IDS = [f"i{i}" for i in range(10)]


@pytest.mark.parametrize("ratios", [(-0.1, 0.6, 0.5), (0.6, -0.1, 0.5), (0.6, 0.5, -0.1)])
def test_each_negative_ratio_is_rejected(ratios):
    with pytest.raises(ValueError, match="non-negative"):
        split_image_ids(IDS, *ratios, seed=0)


def test_zero_ratio_gives_an_empty_part():
    s = split_image_ids(IDS, 0.8, 0.0, 0.2, seed=0)
    assert s["val"] == [] and len(s["train"]) == 8 and len(s["test"]) == 2
    s = split_image_ids(IDS, 0.0, 0.5, 0.5, seed=0)
    assert s["train"] == []


def test_error_messages_name_the_problem():
    with pytest.raises(ValueError, match="unique"):
        split_image_ids(["a", "a"], 0.5, 0.25, 0.25, seed=0)


def test_detection_beside_the_crowd_region_with_y_overlap_is_a_false_positive():
    assert run([det([44, 0, 44.5, 10], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([45, 0, 80, 10])) == pytest.approx(0.5)


def test_detection_above_the_crowd_region_with_x_overlap_is_a_false_positive():
    assert run([det([50, 0, 60, 0.5], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 40, 80, 80])) == pytest.approx(0.5)


def test_short_detection_inside_the_crowd_region_is_ignored():
    assert run([det([50, 50, 60, 50.5], 0.95), det([0, 0, 10, 10], 0.9)], crowd_at([40, 40, 80, 80]), thr=0.75) == 1.0
