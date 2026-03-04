from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

def entropy_from_conf(conf: float, eps: float = 1e-9) -> float:
    # Treat conf as probability of correctness for starter (binary)
    p = np.clip(conf, eps, 1-eps)
    return float(-(p*np.log(p) + (1-p)*np.log(1-p)))

def tta_confidence_std(confs: List[float]) -> float:
    if len(confs) <= 1:
        return 0.0
    return float(np.std(np.array(confs, dtype=float)))
