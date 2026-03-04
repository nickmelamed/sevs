from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sevs.utils.geometry import box_iou, cxcywh

def summarize_consistency(ref_box: np.ndarray, boxes: List[np.ndarray], classes: List[int], confs: List[float], ref_cls: int) -> Dict[str, float]:
    # boxes: matched boxes across probes to ref detection
    if len(boxes) == 0:
        return dict(
            iou_stability=0.0,
            center_jitter=0.0,
            size_jitter=0.0,
            class_churn_rate=1.0,
            confidence_variance=0.0,
        )
    ious = np.array([box_iou(ref_box, b) for b in boxes], dtype=float)
    centers = np.array([cxcywh(b)[:2] for b in boxes], dtype=float)
    sizes = np.array([cxcywh(b)[2:] for b in boxes], dtype=float)
    class_churn = np.mean([1.0 if c != ref_cls else 0.0 for c in classes]) if classes else 0.0
    conf_var = float(np.var(np.array(confs, dtype=float))) if confs else 0.0
    return dict(
        iou_stability=float(np.mean(ious)),
        center_jitter=float(np.mean(np.std(centers, axis=0))) if len(centers) > 1 else 0.0,
        size_jitter=float(np.mean(np.std(sizes, axis=0))) if len(sizes) > 1 else 0.0,
        class_churn_rate=float(class_churn),
        confidence_variance=float(conf_var),
    )
