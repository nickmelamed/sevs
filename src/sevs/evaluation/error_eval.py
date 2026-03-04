from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sevs.utils.metrics import roc_auc

@dataclass
class ErrorDetectionResults:
    auroc: float

def evaluate_error_detection(y_true_incorrect: np.ndarray, y_score_incorrect: np.ndarray) -> ErrorDetectionResults:
    # y_true_incorrect: 1 if incorrect else 0
    return ErrorDetectionResults(auroc=roc_auc(y_true_incorrect, y_score_incorrect))
