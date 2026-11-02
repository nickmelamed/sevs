from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    areaB = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    denom = areaA + areaB - inter
    return float(inter / denom) if denom > 0 else 0.0

def cxcywh(box: np.ndarray) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = box
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    cx = x1 + w/2; cy = y1 + h/2
    return float(cx), float(cy), float(w), float(h)


def area(box: np.ndarray) -> float:
    """Return the area of an ``[x1, y1, x2, y2]`` box, or 0 if it is inverted."""
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def match_detection_to_ground_truth(
    box: np.ndarray,
    cls: int,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
    require_class_match: bool = True,
) -> tuple[bool, float, int]:
    """Return ``(matched, best_iou, gt_index)`` for one detection.

    ``gt_index`` is -1 when no ground-truth box qualifies. Detections are
    matched independently, so two can claim the same ground-truth box.
    """
    best_iou = 0.0
    best_idx = -1
    for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if require_class_match and int(gt_label) != int(cls):
            continue
        iou = box_iou(box, gt_box)
        if best_idx == -1 or iou > best_iou:
            best_iou = iou
            best_idx = j
    return best_iou >= iou_threshold and best_idx != -1, best_iou, best_idx
