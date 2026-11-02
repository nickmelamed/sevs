import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sevs.models.detector import BaseDetector, Detection
from sevs.models.postprocess import PostprocessedDetector, postprocess_detections
from sevs.utils.geometry import box_iou


def det(box, conf, cls=1):
    return Detection(np.array(box, dtype=float), cls, conf)


def test_confidence_threshold_drops_low_scores():
    out = postprocess_detections([det([0, 0, 5, 5], 0.2), det([20, 20, 30, 30], 0.6)], conf_threshold=0.5)
    assert [d.conf for d in out] == [0.6]


def test_nms_keeps_the_higher_score_of_overlapping_same_class_boxes():
    out = postprocess_detections([det([0, 0, 10, 10], 0.6), det([1, 1, 10, 10], 0.9)], nms_iou_threshold=0.5)
    assert [d.conf for d in out] == [0.9]


def test_nms_does_not_suppress_across_classes():
    out = postprocess_detections([det([0, 0, 10, 10], 0.6, cls=1), det([0, 0, 10, 10], 0.9, cls=2)], nms_iou_threshold=0.5)
    assert len(out) == 2


def test_cap_keeps_the_top_scores_sorted():
    dets = [det([i * 20, 0, i * 20 + 10, 10], c) for i, c in enumerate([0.3, 0.9, 0.5])]
    assert [d.conf for d in postprocess_detections(dets, max_detections=2)] == [0.9, 0.5]


def test_none_settings_leave_detections_unchanged_apart_from_order():
    dets = [det([0, 0, 10, 10], 0.1), det([0, 0, 10, 10], 0.2)]
    assert len(postprocess_detections(dets)) == 2


def test_class_filter_keeps_only_listed_classes():
    out = postprocess_detections([det([0, 0, 5, 5], 0.9, cls=1), det([20, 20, 30, 30], 0.8, cls=2)], allowed_classes=[2])
    assert [d.cls for d in out] == [2]


def test_wrapper_postprocesses_every_prediction():
    class Fixed(BaseDetector):
        def predict(self, image):
            return [det([0, 0, 10, 10], 0.2), det([0, 0, 10, 10], 0.8)]

    wrapped = PostprocessedDetector(Fixed(), conf_threshold=0.5)
    assert [d.conf for d in wrapped.predict(np.zeros((5, 5, 3)))] == [0.8]


coord = st.floats(0, 50, allow_nan=False)


@st.composite
def detections(draw):
    x, y = draw(coord), draw(coord)
    return det([x, y, x + draw(st.floats(1, 40)), y + draw(st.floats(1, 40))], draw(st.floats(0.01, 1)), draw(st.integers(1, 2)))


@given(st.lists(detections(), max_size=12), st.floats(0.1, 0.9))
def test_nms_output_has_no_overlapping_pair_in_a_class(dets, thr):
    out = postprocess_detections(dets, nms_iou_threshold=thr)
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if a.cls == b.cls:
                assert box_iou(a.box_xyxy, b.box_xyxy) <= thr


@given(st.lists(detections(), max_size=12), st.floats(0.1, 0.9))
def test_nms_is_idempotent(dets, thr):
    once = postprocess_detections(dets, nms_iou_threshold=thr)
    twice = postprocess_detections(once, nms_iou_threshold=thr)
    assert [(d.cls, d.conf) for d in once] == [(d.cls, d.conf) for d in twice]
