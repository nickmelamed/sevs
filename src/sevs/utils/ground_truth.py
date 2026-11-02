from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sevs.utils.geometry import area


@dataclass
class GroundTruth:
    boxes: np.ndarray
    labels: np.ndarray
    crowd_boxes: np.ndarray
    crowd_labels: np.ndarray


def prepare_ground_truth(
    boxes: np.ndarray,
    labels: np.ndarray,
    meta: dict[str, Any],
    min_box_area: float,
    class_filter: list[int] | None,
    ignore_crowd: bool,
) -> GroundTruth:
    """Apply the ground-truth settings from the config to one image.

    Crowd regions come from ``meta["crowd_boxes_xyxy"]`` and
    ``meta["crowd_labels"]``. They stay separate when ``ignore_crowd`` is set
    and are added to the ordinary boxes otherwise. ``min_box_area`` applies to
    ordinary boxes only.
    """
    crowd_boxes = np.asarray(meta.get("crowd_boxes_xyxy", np.zeros((0, 4))), dtype=float).reshape(-1, 4)
    crowd_labels = np.asarray(meta.get("crowd_labels", np.zeros(0)), dtype=int).reshape(-1)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    keep = np.array([area(b) >= min_box_area for b in boxes], dtype=bool)
    boxes, labels = boxes[keep], labels[keep]
    if not ignore_crowd:
        boxes = np.concatenate([boxes, crowd_boxes])
        labels = np.concatenate([labels, crowd_labels])
        crowd_boxes, crowd_labels = np.zeros((0, 4)), np.zeros(0, dtype=int)
    if class_filter is not None:
        allowed = set(int(c) for c in class_filter)
        keep = np.array([int(c) in allowed for c in labels], dtype=bool)
        boxes, labels = boxes[keep], labels[keep]
        keep = np.array([int(c) in allowed for c in crowd_labels], dtype=bool)
        crowd_boxes, crowd_labels = crowd_boxes[keep], crowd_labels[keep]
    return GroundTruth(boxes, labels, crowd_boxes, crowd_labels)
