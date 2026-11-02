"""COCO-style detection AP.

Matching, crowd handling, and the 101-point recall interpolation follow
pycocotools' ``COCOeval`` for the ``bbox`` task with the "all" area range, so
the numbers can be compared with published COCO results. Precision is
tp / (tp + fp) without COCOeval's 1e-16 guard term, so results agree with
pycocotools to about 1e-15 in the parity tests, including tied scores.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sevs.models.detector import Detection
from sevs.utils.geometry import box_ioa, box_iou

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


@dataclass
class DetectionEvalResults:
    map50: float | None
    map50_95: float | None
    per_class_ap50: dict[int, float]
    backend: str


def _image_order(image_id: str) -> tuple[int, int, str]:
    """Order numeric ids by value, as pycocotools does, and other ids by text after them."""
    text = str(image_id)
    return (0, int(text), text) if text.lstrip("-").isdigit() else (1, 0, text)


def _match_image_class(
    det_boxes: np.ndarray, gt_boxes: np.ndarray, crowd_boxes: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(matched, ignored)`` per detection, detections already sorted by score."""
    n_det, n_gt, n_crowd = len(det_boxes), len(gt_boxes), len(crowd_boxes)
    matched = np.zeros(n_det, dtype=bool)
    ignored = np.zeros(n_det, dtype=bool)
    if n_det == 0 or n_gt + n_crowd == 0:
        return matched, ignored
    ious = np.zeros((n_det, n_gt + n_crowd))
    for d in range(n_det):
        for g in range(n_gt):
            ious[d, g] = box_iou(det_boxes[d], gt_boxes[g])
        for c in range(n_crowd):
            ious[d, n_gt + c] = box_ioa(det_boxes[d], crowd_boxes[c])
    is_crowd = np.array([False] * n_gt + [True] * n_crowd)
    taken = np.zeros(n_gt + n_crowd, dtype=bool)
    for d in range(n_det):
        best_iou = min(iou_thr, 1 - 1e-10)
        best = -1
        for g in range(n_gt + n_crowd):
            if taken[g] and not is_crowd[g]:
                continue
            if best > -1 and not is_crowd[best] and is_crowd[g]:
                break
            if ious[d, g] < best_iou:
                continue
            best_iou = ious[d, g]
            best = g
        if best == -1:
            continue
        ignored[d] = is_crowd[best]
        matched[d] = True
        taken[best] = True
    return matched, ignored


def _average_precision(scores: np.ndarray, matched: np.ndarray, ignored: np.ndarray, n_gt: int) -> float:
    order = np.argsort(-scores, kind="mergesort")
    tps = (matched & ~ignored)[order]
    fps = (~matched & ~ignored)[order]
    tp_sum = np.cumsum(tps).astype(float)
    fp_sum = np.cumsum(fps).astype(float)
    if len(tp_sum) == 0:
        return 0.0
    recall = tp_sum / n_gt
    denom = fp_sum + tp_sum
    precision = np.divide(tp_sum, denom, out=np.zeros_like(tp_sum), where=denom > 0).tolist()
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]
    idx = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    q = np.zeros(len(RECALL_THRESHOLDS))
    for ri, pi in enumerate(idx):
        if pi < len(precision):
            q[ri] = precision[pi]
    return float(np.mean(q))


def evaluate_detection_predictions(
    predictions: Sequence[tuple[str, list[Detection]]],
    ground_truths: Sequence[tuple[str, np.ndarray, np.ndarray]],
    iou_thresholds: Sequence[float] | None = None,
    crowd: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    max_dets: int = 100,
) -> DetectionEvalResults:
    """Return mAP@0.50 and mAP@0.50:0.95 in the style of COCOeval.

    ``crowd`` maps an image id to ``(boxes, labels)`` of crowd regions.
    Detections that fall on a crowd region of their class are ignored instead
    of counted as false positives. Only the ``max_dets`` highest-scoring
    detections of each class in each image are used.

    A class with no non-crowd ground truth is skipped, and the result is NaN
    if every class is skipped.
    """
    thresholds = list(iou_thresholds) if iou_thresholds is not None else IOU_THRESHOLDS
    crowd = crowd or {}
    class_ids = sorted({int(c) for _, _, labels in ground_truths for c in labels.tolist()})
    if not class_ids:
        return DetectionEvalResults(map50=float("nan"), map50_95=float("nan"), per_class_ap50={}, backend="coco-style")
    gt_by_image = {image_id: (boxes, labels) for image_id, boxes, labels in ground_truths}
    pred_by_image = {image_id: dets for image_id, dets in predictions}
    image_ids = sorted(set(gt_by_image) | set(pred_by_image) | set(crowd), key=_image_order)

    per_class_ap: dict[int, list[float]] = {}
    for cls in class_ids:
        n_gt = sum(int(np.sum(labels == cls)) for _, labels in gt_by_image.values())
        scores: list[float] = []
        flags: list[list[tuple[bool, bool]]] = [[] for _ in thresholds]
        for image_id in image_ids:
            boxes, labels = gt_by_image.get(image_id, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
            gt_boxes = boxes[labels == cls]
            cb, cl = crowd.get(image_id, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
            crowd_boxes = cb[cl == cls]
            dets = [d for d in pred_by_image.get(image_id, []) if d.cls == cls]
            dets.sort(key=lambda d: -d.conf)
            dets = dets[:max_dets]
            if not dets:
                continue
            det_boxes = np.array([d.box_xyxy for d in dets])
            scores.extend(d.conf for d in dets)
            for k, thr in enumerate(thresholds):
                matched, ignored = _match_image_class(det_boxes, gt_boxes, crowd_boxes, thr)
                flags[k].extend(zip(matched.tolist(), ignored.tolist()))
        if n_gt == 0:
            continue
        aps = []
        for k in range(len(thresholds)):
            matched = np.array([m for m, _ in flags[k]], dtype=bool)
            ignored = np.array([i for _, i in flags[k]], dtype=bool)
            aps.append(_average_precision(np.array(scores), matched, ignored, n_gt))
        per_class_ap[cls] = aps

    if not per_class_ap:
        return DetectionEvalResults(map50=float("nan"), map50_95=float("nan"), per_class_ap50={}, backend="coco-style")
    k50 = thresholds.index(0.5) if 0.5 in thresholds else None
    per_class_ap50 = {c: aps[k50] for c, aps in per_class_ap.items()} if k50 is not None else {}
    return DetectionEvalResults(
        map50=float(np.mean(list(per_class_ap50.values()))) if k50 is not None else float("nan"),
        map50_95=float(np.mean([a for aps in per_class_ap.values() for a in aps])),
        per_class_ap50=per_class_ap50,
        backend="coco-style",
    )
