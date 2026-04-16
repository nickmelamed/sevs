from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from sevs.models.detector import Detection
from sevs.utils.geometry import box_iou


@dataclass
class DetectionEvalResults:
    map50: float | None
    map50_95: float | None
    per_class_ap50: Dict[int, float]
    backend: str


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_at_iou(preds, gts, class_id: int, iou_thr: float) -> float:
    npos = 0
    class_gts = {}
    for image_id, gt_boxes, gt_labels in gts:
        mask = gt_labels == class_id
        boxes = gt_boxes[mask]
        npos += len(boxes)
        class_gts[image_id] = {"boxes": boxes, "matched": np.zeros(len(boxes), dtype=bool)}

    class_preds = []
    for image_id, detections in preds:
        for d in detections:
            if d.cls == class_id:
                class_preds.append((image_id, float(d.conf), d.box_xyxy))
    class_preds.sort(key=lambda x: x[1], reverse=True)
    if npos == 0:
        return float("nan")
    tp = np.zeros(len(class_preds), dtype=float)
    fp = np.zeros(len(class_preds), dtype=float)
    for i, (image_id, _score, box) in enumerate(class_preds):
        gt = class_gts.get(image_id, {"boxes": np.zeros((0,4)), "matched": np.zeros((0,), dtype=bool)})
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt["boxes"]):
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0 and not gt["matched"][best_j]:
            tp[i] = 1.0
            gt["matched"][best_j] = True
        else:
            fp[i] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(npos, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _voc_ap(rec, prec)


def evaluate_detection_predictions(
    predictions: Sequence[tuple[str, List[Detection]]],
    ground_truths: Sequence[tuple[str, np.ndarray, np.ndarray]],
    iou_thresholds: Iterable[float] | None = None,
) -> DetectionEvalResults:
    iou_thresholds = list(iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    class_ids = sorted({int(c) for _, _, labels in ground_truths for c in labels.tolist()})
    per_class_ap50 = {}
    if not class_ids:
        return DetectionEvalResults(map50=float("nan"), map50_95=float("nan"), per_class_ap50={}, backend="native")
    ap50s = []
    maps = []
    for cls in class_ids:
        ap50 = _ap_at_iou(predictions, ground_truths, cls, 0.5)
        per_class_ap50[cls] = ap50
        if not np.isnan(ap50):
            ap50s.append(ap50)
        cls_aps = []
        for thr in iou_thresholds:
            ap = _ap_at_iou(predictions, ground_truths, cls, float(thr))
            if not np.isnan(ap):
                cls_aps.append(ap)
        if cls_aps:
            maps.append(float(np.mean(cls_aps)))
    return DetectionEvalResults(
        map50=float(np.mean(ap50s)) if ap50s else float("nan"),
        map50_95=float(np.mean(maps)) if maps else float("nan"),
        per_class_ap50=per_class_ap50,
        backend="native",
    )

