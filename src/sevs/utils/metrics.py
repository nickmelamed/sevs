from __future__ import annotations
from typing import Tuple
import numpy as np

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # Binary ECE for "is_correct" with prob=confidence (or meta prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / len(y_prob)) * abs(acc - conf)
    return float(ece)

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Minimal AUROC without sklearn (but we do have sklearn; keeping standalone)
    order = np.argsort(-y_score)
    y = y_true[order].astype(int)
    pos = np.sum(y == 1); neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / pos
    fpr = fps / neg
    # trapezoidal
    return float(np.trapezoid(tpr, fpr))
