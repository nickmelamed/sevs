import numpy as np

from sevs.utils.ground_truth import prepare_ground_truth

BOXES = np.array([[0, 0, 10, 10], [0, 0, 2, 2], [20, 20, 40, 40]], dtype=float)
LABELS = np.array([1, 1, 2])
CROWD = {"crowd_boxes_xyxy": np.array([[50, 50, 90, 90]], dtype=float), "crowd_labels": np.array([1])}


def test_min_box_area_drops_small_boxes():
    gt = prepare_ground_truth(BOXES, LABELS, {}, min_box_area=10, class_filter=None, ignore_crowd=True)
    assert gt.labels.tolist() == [1, 2]


def test_class_filter_applies_to_ground_truth_and_crowd():
    gt = prepare_ground_truth(BOXES, LABELS, CROWD, min_box_area=0, class_filter=[2], ignore_crowd=True)
    assert gt.labels.tolist() == [2]
    assert len(gt.crowd_boxes) == 0


def test_ignored_crowd_is_kept_apart_from_ground_truth():
    gt = prepare_ground_truth(BOXES, LABELS, CROWD, min_box_area=0, class_filter=None, ignore_crowd=True)
    assert len(gt.boxes) == 3 and len(gt.crowd_boxes) == 1


def test_crowd_becomes_ordinary_ground_truth_when_not_ignored():
    gt = prepare_ground_truth(BOXES, LABELS, CROWD, min_box_area=0, class_filter=None, ignore_crowd=False)
    assert len(gt.boxes) == 4 and len(gt.crowd_boxes) == 0


def test_missing_crowd_meta_is_empty():
    gt = prepare_ground_truth(BOXES, LABELS, {}, min_box_area=0, class_filter=None, ignore_crowd=True)
    assert gt.crowd_boxes.shape == (0, 4) and gt.crowd_labels.shape == (0,)
