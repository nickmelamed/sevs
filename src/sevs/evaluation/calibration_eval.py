from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sevs.utils.metrics import expected_calibration_error, brier_score

@dataclass
class CalibrationResults:
    ece: float
    brier: float

def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> CalibrationResults:
    return CalibrationResults(
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        brier=brier_score(y_true, y_prob),
    )
