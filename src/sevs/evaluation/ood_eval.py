from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from sevs.models.detector import Detection
from sevs.models.uncertainty import entropy_from_conf


@dataclass
class OODScoreSummary:
    method: str
    image_scores: List[float]
    mean_score: float


def _energy_from_logits(logits: np.ndarray, temperature: float = 1.0) -> float:
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    m = np.max(z)
    return float(-temperature * (m + np.log(np.sum(np.exp(z - m)))))


def detection_level_scores(detections: List[Detection], method: str = "max_softmax_prob") -> List[float]:
    scores: List[float] = []
    for d in detections:
        if method == "max_softmax_prob":
            scores.append(1.0 - float(d.conf))
        elif method == "entropy":
            if d.logits is not None:
                p = np.exp(d.logits - np.max(d.logits))
                p = p / np.sum(p)
                scores.append(float(-(p * np.log(np.clip(p, 1e-9, 1.0))).sum()))
            else:
                scores.append(entropy_from_conf(float(d.conf)))
        elif method == "energy":
            if d.logits is not None:
                scores.append(_energy_from_logits(d.logits))
            else:
                p = np.clip(float(d.conf), 1e-6, 1 - 1e-6)
                pseudo_logits = np.log(np.array([p, 1 - p]))
                scores.append(_energy_from_logits(pseudo_logits))
        else:
            raise ValueError(f"Unknown OOD scoring method: {method}")
    return scores


def summarize_ood_scores(images_to_detections: Iterable[List[Detection]], method: str = "max_softmax_prob", reduction: str = "max") -> OODScoreSummary:
    image_scores: List[float] = []
    for detections in images_to_detections:
        det_scores = detection_level_scores(detections, method=method)
        if not det_scores:
            image_scores.append(1.0)
            continue
        if reduction == "max":
            image_scores.append(float(np.max(det_scores)))
        elif reduction == "mean":
            image_scores.append(float(np.mean(det_scores)))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    return OODScoreSummary(method=method, image_scores=image_scores, mean_score=float(np.mean(image_scores) if image_scores else float("nan")))
