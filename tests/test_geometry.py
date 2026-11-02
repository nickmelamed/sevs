import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sevs.utils.geometry import area, box_iou, cxcywh, match_detection_to_ground_truth

coord = st.floats(0, 100, allow_nan=False, allow_infinity=False)


@st.composite
def boxes(draw):
    x1, y1 = draw(coord), draw(coord)
    w = draw(st.floats(0.5, 50))
    h = draw(st.floats(0.5, 50))
    return np.array([x1, y1, x1 + w, y1 + h])


def test_iou_known_values():
    a = np.array([0, 0, 10, 10], dtype=float)
    assert box_iou(a, a) == 1.0
    assert box_iou(a, np.array([20, 20, 30, 30], dtype=float)) == 0.0
    assert box_iou(a, np.array([5, 0, 15, 10], dtype=float)) == pytest.approx(50 / 150)


def test_iou_degenerate_boxes_are_zero():
    z = np.zeros(4)
    assert box_iou(z, z) == 0.0


@given(boxes(), boxes())
def test_iou_symmetric_and_bounded(a, b):
    assert box_iou(a, b) == pytest.approx(box_iou(b, a))
    assert 0.0 <= box_iou(a, b) <= 1.0 + 1e-12


def test_cxcywh():
    assert cxcywh(np.array([0, 0, 10, 20], dtype=float)) == (5.0, 10.0, 10.0, 20.0)


def test_area():
    assert area(np.array([1, 2, 4, 6], dtype=float)) == 12.0
    assert area(np.array([4, 6, 1, 2], dtype=float)) == 0.0


@given(boxes())
def test_area_nonnegative(b):
    assert area(b) >= 0.0


GT = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
LABELS = np.array([1, 2])


def test_match_hit():
    box = np.array([1, 1, 10, 10], dtype=float)
    matched, iou, idx = match_detection_to_ground_truth(box, 1, GT, LABELS, 0.5, True)
    assert matched and idx == 0 and iou == pytest.approx(81 / 100)


def test_match_wrong_class_is_miss_when_class_required():
    box = np.array([0, 0, 10, 10], dtype=float)
    matched, iou, idx = match_detection_to_ground_truth(box, 3, GT, LABELS, 0.5, True)
    assert not matched and idx == -1 and iou == 0.0


def test_match_ignores_class_when_not_required():
    box = np.array([0, 0, 10, 10], dtype=float)
    matched, _, idx = match_detection_to_ground_truth(box, 2, GT, LABELS, 0.5, False)
    assert matched and idx == 0


def test_match_below_threshold():
    box = np.array([0, 0, 10, 10], dtype=float)
    matched, iou, idx = match_detection_to_ground_truth(
        box, 1, np.array([[5, 0, 15, 10]], dtype=float), np.array([1]), 0.5, True
    )
    assert not matched and idx == 0 and iou == pytest.approx(1 / 3)


def test_match_without_ground_truth():
    box = np.array([0, 0, 10, 10], dtype=float)
    out = match_detection_to_ground_truth(box, 1, np.zeros((0, 4)), np.zeros((0,), dtype=int), 0.5, True)
    assert out == (False, 0.0, -1)


def test_two_detections_can_match_the_same_ground_truth_box():
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    labels = np.array([1])
    first = match_detection_to_ground_truth(np.array([0, 0, 10, 10], dtype=float), 1, gt, labels, 0.5, True)
    second = match_detection_to_ground_truth(np.array([1, 1, 10, 10], dtype=float), 1, gt, labels, 0.5, True)
    assert first[0] and second[0] and first[2] == second[2] == 0
