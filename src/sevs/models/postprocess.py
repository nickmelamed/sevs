from __future__ import annotations

import numpy as np

from sevs.models.detector import BaseDetector, Detection
from sevs.utils.geometry import box_iou


def postprocess_detections(
    dets: list[Detection],
    conf_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
    max_detections: int | None = None,
    allowed_classes: list[int] | None = None,
) -> list[Detection]:
    """Filter by class and confidence, apply class-wise NMS, and keep the top scores.

    The result is sorted by confidence, highest first. A setting of ``None``
    skips that step.
    """
    allowed = None if allowed_classes is None else {int(c) for c in allowed_classes}
    kept = [
        d
        for d in dets
        if (conf_threshold is None or d.conf >= conf_threshold) and (allowed is None or d.cls in allowed)
    ]
    kept.sort(key=lambda d: -d.conf)
    if nms_iou_threshold is not None:
        survivors: list[Detection] = []
        for d in kept:
            if all(
                s.cls != d.cls or box_iou(np.asarray(s.box_xyxy), np.asarray(d.box_xyxy)) <= nms_iou_threshold
                for s in survivors
            ):
                survivors.append(d)
        kept = survivors
    return kept[:max_detections] if max_detections is not None else kept


class PostprocessedDetector(BaseDetector):
    """Apply ``postprocess_detections`` to every prediction of ``detector``."""

    def __init__(
        self,
        detector: BaseDetector,
        conf_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
        max_detections: int | None = None,
        allowed_classes: list[int] | None = None,
    ):
        self.detector = detector
        self.allowed_classes = allowed_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def predict(self, image: np.ndarray) -> list[Detection]:
        return postprocess_detections(
            self.detector.predict(image), self.conf_threshold,
            self.nms_iou_threshold,
            self.max_detections,
            self.allowed_classes,
        )
