from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

@dataclass
class Detection:
    box_xyxy: np.ndarray   # [x1,y1,x2,y2]
    cls: int
    conf: float

class BaseDetector:
    def predict(self, image: np.ndarray) -> List[Detection]:
        raise NotImplementedError

class MockDetector(BaseDetector):
    """A detector that returns deterministic pseudo-detections.
    This lets you validate the entire SEVS pipeline without a dataset.
    Replace with a real wrapper (YOLO/DETR/etc.) when ready.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, image: np.ndarray) -> List[Detection]:
        H, W = image.shape[:2]
        n = int(self.rng.integers(1, 6))
        dets: List[Detection] = []
        for _ in range(n):
            x1 = float(self.rng.uniform(0, W*0.7))
            y1 = float(self.rng.uniform(0, H*0.7))
            x2 = float(min(W, x1 + self.rng.uniform(W*0.05, W*0.3)))
            y2 = float(min(H, y1 + self.rng.uniform(H*0.05, H*0.3)))
            cls = int(self.rng.integers(0, 5))
            conf = float(self.rng.uniform(0.05, 0.99))
            dets.append(Detection(np.array([x1,y1,x2,y2], dtype=float), cls, conf))
        return dets

def build_detector(cfg: Dict[str, Any], seed: int = 42) -> BaseDetector:
    family = cfg.get("family", "mock")
    if family == "mock":
        return MockDetector(seed=seed)
    raise NotImplementedError(
        f"Detector family '{family}' not implemented in starter. "
        "Plug your detector wrapper into src/sevs/models/detector.py."
    )
