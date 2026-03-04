import numpy as np
from sevs.utils.metrics import expected_calibration_error, brier_score

def test_brier():
    y = np.array([0,1,1,0])
    p = np.array([0.1,0.9,0.8,0.2])
    assert 0 <= brier_score(y,p) <= 1

def test_ece():
    y = np.array([0,1,1,0])
    p = np.array([0.1,0.9,0.8,0.2])
    e = expected_calibration_error(y,p,n_bins=2)
    assert 0 <= e <= 1
