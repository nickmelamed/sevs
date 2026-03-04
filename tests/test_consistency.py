import numpy as np
from sevs.probes.consistency import summarize_consistency

def test_consistency_keys():
    ref = np.array([0,0,10,10], dtype=float)
    boxes = [ref.copy(), ref.copy()]
    classes = [1,1]
    confs = [0.8,0.9]
    d = summarize_consistency(ref, boxes, classes, confs, ref_cls=1)
    assert "iou_stability" in d and "confidence_variance" in d
