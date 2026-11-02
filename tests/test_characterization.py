"""Pin current behavior of the core numeric code with small synthetic inputs."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sevs.evaluation.calibration_eval import evaluate_calibration
from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.evaluation.ood_eval import detection_level_scores, summarize_ood_scores
from sevs.models.detector import Detection, MockDetector
from sevs.models.meta_error import build_meta_model
from sevs.models.uncertainty import entropy_from_conf, tta_confidence_std
from sevs.probes.consistency import summarize_consistency
from sevs.probes.perturbations import apply_perturbation_with_inverse
from sevs.utils.metrics import brier_score, expected_calibration_error

probs = st.floats(0, 1, allow_nan=False)


def det(box, cls=1, conf=0.9, logits=None):
    return Detection(np.array(box, dtype=float), cls, conf, logits)


# calibration

def test_ece_and_brier_hand_worked():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.1)
    assert brier_score(y, p) == pytest.approx(0.01)


def test_ece_zero_when_confidence_matches_accuracy():
    y = np.array([1, 0, 1, 0])
    p = np.full(4, 0.5)
    assert expected_calibration_error(y, p, n_bins=5) == pytest.approx(0.0)


def test_ece_counts_probability_one_in_last_bin():
    assert expected_calibration_error(np.array([0]), np.array([1.0]), n_bins=4) == pytest.approx(1.0)


def test_evaluate_calibration_wraps_metrics():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    out = evaluate_calibration(y, p, n_bins=5)
    assert out.ece == pytest.approx(expected_calibration_error(y, p, n_bins=5))
    assert out.brier == pytest.approx(brier_score(y, p))


@given(st.lists(st.tuples(st.integers(0, 1), probs), min_size=1, max_size=50), st.integers(1, 20))
def test_ece_and_brier_bounded(pairs, bins):
    y = np.array([a for a, _ in pairs])
    p = np.array([b for _, b in pairs])
    assert 0.0 <= expected_calibration_error(y, p, n_bins=bins) <= 1.0 + 1e-9
    assert 0.0 <= brier_score(y, p) <= 1.0 + 1e-9


# detection mAP

def test_map_no_predictions_is_zero():
    gts = [("a", np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]
    out = evaluate_detection_predictions([("a", [])], gts, iou_thresholds=[0.5])
    assert out.map50 == 0.0


def test_map_false_positive_ranked_first_halves_ap():
    gts = [("a", np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]
    preds = [("a", [det([50, 50, 60, 60], conf=0.9), det([0, 0, 10, 10], conf=0.5)])]
    assert evaluate_detection_predictions(preds, gts, iou_thresholds=[0.5]).map50 == pytest.approx(0.5)


def test_map_duplicate_detection_counts_as_false_positive():
    gts = [("a", np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]
    preds = [("a", [det([0, 0, 10, 10], conf=0.9), det([0, 0, 10, 10], conf=0.8)])]
    assert evaluate_detection_predictions(preds, gts, iou_thresholds=[0.5]).map50 == 1.0


def test_map_without_ground_truth_is_nan():
    out = evaluate_detection_predictions([("a", [det([0, 0, 1, 1])])], [("a", np.zeros((0, 4)), np.zeros((0,), dtype=int))])
    assert np.isnan(out.map50) and np.isnan(out.map50_95)


def test_map_averages_over_classes():
    gts = [("a", np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float), np.array([1, 2]))]
    preds = [("a", [det([0, 0, 10, 10], cls=1)])]
    out = evaluate_detection_predictions(preds, gts, iou_thresholds=[0.5])
    assert out.per_class_ap50 == {1: 1.0, 2: 0.0}
    assert out.map50 == pytest.approx(0.5)


def test_map5095_is_below_map50_for_loose_boxes():
    gts = [("a", np.array([[0, 0, 10, 10]], dtype=float), np.array([1]))]
    preds = [("a", [det([0, 0, 10, 8])])]
    out = evaluate_detection_predictions(preds, gts)
    assert out.map50 == 1.0
    assert 0.0 < out.map50_95 < 1.0


# consistency

def test_consistency_without_matches_is_worst_case():
    out = summarize_consistency(np.array([0, 0, 10, 10], dtype=float), [], [], [], ref_cls=1)
    assert out == dict(iou_stability=0.0, center_jitter=0.0, size_jitter=0.0, class_churn_rate=1.0, confidence_variance=0.0)


def test_consistency_identical_probes_are_stable():
    ref = np.array([0, 0, 10, 10], dtype=float)
    out = summarize_consistency(ref, [ref, ref], [1, 1], [0.8, 0.8], ref_cls=1)
    assert out["iou_stability"] == 1.0
    assert out["center_jitter"] == 0.0 and out["class_churn_rate"] == 0.0 and out["confidence_variance"] == 0.0


def test_consistency_class_churn_and_confidence_variance():
    ref = np.array([0, 0, 10, 10], dtype=float)
    out = summarize_consistency(ref, [ref, ref], [1, 2], [0.5, 0.7], ref_cls=1)
    assert out["class_churn_rate"] == 0.5
    assert out["confidence_variance"] == pytest.approx(0.01)


# uncertainty

def test_entropy_from_conf_peaks_at_half():
    assert entropy_from_conf(0.5) == pytest.approx(np.log(2))
    assert entropy_from_conf(0.0) == pytest.approx(0.0, abs=1e-6)


@given(probs)
def test_entropy_symmetric_and_bounded(p):
    assert entropy_from_conf(p) == pytest.approx(entropy_from_conf(1 - p), abs=1e-6)
    assert 0.0 <= entropy_from_conf(p) <= np.log(2) + 1e-9


def test_tta_std_needs_two_values():
    assert tta_confidence_std([0.9]) == 0.0
    assert tta_confidence_std([0.5, 0.7]) == pytest.approx(0.1)


# OOD

def test_ood_max_softmax_score_is_one_minus_conf():
    assert detection_level_scores([det([0, 0, 1, 1], conf=0.8)], "max_softmax_prob") == pytest.approx([0.2])


def test_ood_energy_from_logits():
    d = det([0, 0, 1, 1], logits=np.array([0.0, 0.0]))
    assert detection_level_scores([d], "energy") == pytest.approx([-np.log(2)])


def test_ood_unknown_method_raises():
    with pytest.raises(ValueError):
        detection_level_scores([det([0, 0, 1, 1])], "nope")


def test_ood_image_without_detections_scores_one():
    out = summarize_ood_scores([[], [det([0, 0, 1, 1], conf=0.6)]], "max_softmax_prob")
    assert out.image_scores == pytest.approx([1.0, 0.4])
    assert out.mean_score == pytest.approx(0.7)


# perturbations

@given(st.sampled_from(["hflip", "random_crop"]), st.integers(1, 3))
def test_geometric_perturbations_keep_image_and_box_shapes(name, severity):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    res = apply_perturbation_with_inverse(img, name, severity)
    box = np.array([[10.0, 10.0, 30.0, 25.0]])
    assert res.image.shape == img.shape
    assert res.invert_boxes(box).shape == box.shape


def test_random_crop_inverse_undoes_the_forward_mapping():
    h, w, severity = 40, 60, 2
    pad = int(min(h, w) * 0.05 * severity)
    sx, sy = (w - 2 * pad) / w, (h - 2 * pad) / h
    original = np.array([[20.0, 15.0, 40.0, 30.0]])
    forward = original.copy()
    forward[:, [0, 2]] = (original[:, [0, 2]] - pad) / sx
    forward[:, [1, 3]] = (original[:, [1, 3]] - pad) / sy
    res = apply_perturbation_with_inverse(np.zeros((h, w, 3), dtype=np.uint8), "random_crop", severity)
    assert res.invert_boxes(forward) == pytest.approx(original)


def test_hflip_inverse_mirrors_boxes():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    res = apply_perturbation_with_inverse(img, "hflip", 1)
    out = res.invert_boxes(np.array([[2.0, 1.0, 5.0, 4.0]]))
    assert out.tolist() == [[15.0, 1.0, 18.0, 4.0]]


def test_non_geometric_perturbations_keep_shape_and_boxes():
    img = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    box = np.array([[1.0, 2.0, 3.0, 4.0]])
    for name in ["gaussian_blur", "brightness", "contrast", "jpeg_compress"]:
        res = apply_perturbation_with_inverse(img, name, 2)
        assert res.image.shape == img.shape and res.image.dtype == np.uint8
        assert res.invert_boxes(box).tolist() == box.tolist()


def test_unknown_perturbation_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(apply_perturbation_with_inverse(img, "nope", 1).image, img)


# detector and meta-model

def test_mock_detector_is_deterministic_for_a_seed():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    a = MockDetector(seed=3).predict(img)
    b = MockDetector(seed=3).predict(img)
    assert [(d.cls, d.conf) for d in a] == [(d.cls, d.conf) for d in b]
    assert 1 <= len(a) <= 5


def test_meta_model_returns_probability_of_incorrect():
    cfg = {"model": {"family": "logistic_regression", "params": {"C": 1.0, "max_iter": 200}}}
    m = build_meta_model(cfg, ["f"])
    x = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])
    m.fit(x, y)
    p = m.predict_proba(x)
    assert p.shape == (4,) and p[0] < 0.5 < p[3]


def test_meta_model_unknown_family_raises():
    with pytest.raises(NotImplementedError):
        build_meta_model({"model": {"family": "mlp"}}, [])
