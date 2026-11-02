import numpy as np
import pytest

from sevs.evaluation.error_eval import abstention_tradeoff, evaluate_error_detection


def test_auprc_hand_worked():
    y = np.array([1, 0, 1])
    s = np.array([0.9, 0.8, 0.7])
    out = evaluate_error_detection(y, s)
    assert out.auprc == pytest.approx((1 + 2 / 3) / 2)
    assert out.auroc == pytest.approx(0.5)


def test_single_class_gives_nan():
    out = evaluate_error_detection(np.array([0, 0]), np.array([0.1, 0.2]))
    assert np.isnan(out.auroc) and np.isnan(out.auprc)


def test_abstention_coverage_and_risk():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.9])
    rows = abstention_tradeoff(y, p, [0.15, 0.5, 1.0])
    assert rows[0] == (0.15, 0.25, 0.0)
    assert rows[1] == (0.5, 0.5, 0.0)
    assert rows[2] == (1.0, 1.0, 0.5)


def test_abstention_with_nothing_accepted_has_nan_risk():
    (t, cov, risk), = abstention_tradeoff(np.array([1]), np.array([0.9]), [0.1])
    assert cov == 0.0 and np.isnan(risk)
