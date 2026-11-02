from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from sevs.utils.metrics import roc_auc


@dataclass
class ErrorDetectionResults:
    auroc: float
    auprc: float


def evaluate_error_detection(y_true_incorrect: np.ndarray, y_score_incorrect: np.ndarray) -> ErrorDetectionResults:
    """Score how well ``y_score_incorrect`` ranks incorrect detections (label 1) first."""
    y = np.asarray(y_true_incorrect).astype(int)
    if y.min() == y.max():
        return ErrorDetectionResults(auroc=float("nan"), auprc=float("nan"))
    return ErrorDetectionResults(
        auroc=roc_auc(y, y_score_incorrect),
        auprc=float(average_precision_score(y, y_score_incorrect)),
    )


def abstention_tradeoff(
    y_incorrect: np.ndarray, p_incorrect: np.ndarray, thresholds: list[float]
) -> list[tuple[float, float, float]]:
    """Return ``(threshold, coverage, risk)`` for accepting detections with P(incorrect) <= threshold.

    Coverage is the accepted fraction. Risk is the error rate among accepted
    detections, and NaN when none are accepted.
    """
    y = np.asarray(y_incorrect, dtype=float)
    p = np.asarray(p_incorrect, dtype=float)
    out = []
    for t in thresholds:
        accepted = p <= t
        coverage = float(np.mean(accepted)) if len(p) else float("nan")
        risk = float(np.mean(y[accepted])) if accepted.any() else float("nan")
        out.append((float(t), coverage, risk))
    return out
