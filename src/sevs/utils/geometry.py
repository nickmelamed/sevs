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
