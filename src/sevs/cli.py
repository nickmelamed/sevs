from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from sevs.datasets.registry import get_dataset
from sevs.evaluation.calibration_eval import evaluate_calibration
from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.evaluation.error_eval import evaluate_error_detection
from sevs.evaluation.ood_eval import summarize_ood_scores
from sevs.evaluation.report import save_reliability_diagram, save_results_table
from sevs.logging import get_logger
from sevs.models.detector import Detection, build_detector
from sevs.models.meta_error import build_meta_model
from sevs.models.uncertainty import entropy_from_conf, tta_confidence_std
from sevs.probes.consistency import summarize_consistency
from sevs.probes.perturbations import apply_perturbation_with_inverse
from sevs.probes.tta import generate_tta_images
from sevs.utils.geometry import area, box_iou, match_detection_to_ground_truth
from sevs.utils.io import ensure_dir, load_yaml
from sevs.utils.seed import seed_everything

logger = get_logger()


def _mock_image(seed: int = 0, H: int = 480, W: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def _iter_samples(cfg: Dict[str, Any]):
    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "mock":
        for i in range(int(cfg["dataset"].get("max_items", 12))):
            yield {
                "image": _mock_image(int(cfg["run"]["seed"]) + i),
                "image_id": f"mock_{i}",
                "gt_boxes_xyxy": np.zeros((0, 4), dtype=float),
                "gt_labels": np.zeros((0,), dtype=int),
                "meta": {},
            }
        return
    ds_cls = get_dataset(dataset_name)
    ds = ds_cls(cfg["dataset"])
    for sample in ds.iter_samples():
        yield {
            "image": sample.image,
            "image_id": sample.image_id,
            "gt_boxes_xyxy": sample.gt_boxes_xyxy,
            "gt_labels": sample.gt_labels,
            "meta": sample.meta,
        }


def _match_reference_to_probe(ref_det: Detection, probe_dets: List[Detection], class_match_required: bool = True) -> tuple[list[np.ndarray], list[int], list[float]]:
    best_iou = 0.0
    best: Detection | None = None
    for d in probe_dets:
        if class_match_required and d.cls != ref_det.cls:
            continue
        iou = box_iou(ref_det.box_xyxy, d.box_xyxy)
        if iou > best_iou:
            best_iou = iou
            best = d
    if best is None:
        return [], [], []
    return [best.box_xyxy], [best.cls], [best.conf]


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg["run"]["seed"])
    seed_everything(seed)

    out_dir = cfg["run"]["output_dir"]
    ensure_dir(out_dir)

    device = cfg.get("run", {}).get("device", {}).get("backend")
    detector = build_detector(cfg["detector"], seed=seed, device=device)
    feature_names = cfg["meta_error_predictor"]["features"]
    iou_threshold = float(cfg["ground_truth"].get("match_iou_threshold", 0.5))
    require_class_match = bool(cfg["probes"].get("matching", {}).get("class_match_required", True))

    calibration_true: List[int] = []
    calibration_prob: List[float] = []
    X_rows: List[List[float]] = []
    y_incorrect: List[int] = []
    prediction_records: List[tuple[str, List[Detection]]] = []
    gt_records: List[tuple[str, np.ndarray, np.ndarray]] = []
    ood_detections: List[List[Detection]] = []

    total_images = 0
    total_detections = 0

    for sample in _iter_samples(cfg):
        total_images += 1
        image = sample["image"]
        image_id = sample["image_id"]
        gt_boxes = sample["gt_boxes_xyxy"]
        gt_labels = sample["gt_labels"]

        dets = detector.predict(image)
        prediction_records.append((image_id, dets))
        gt_records.append((image_id, gt_boxes, gt_labels))
        ood_detections.append(dets)
        total_detections += len(dets)

        perturbations = cfg.get("probes", {}).get("perturbations", [])
        probe_predictions: List[tuple[List[Detection], Any]] = []
        for p in perturbations:
            res = apply_perturbation_with_inverse(image, p["name"], int(p.get("severity", 1)))
            probe_dets = detector.predict(res.image)
            if probe_dets:
                inv_boxes = res.invert_boxes(np.stack([d.box_xyxy for d in probe_dets], axis=0))
                for d, inv_box in zip(probe_dets, inv_boxes):
                    d.box_xyxy = inv_box
            probe_predictions.append((probe_dets, res.invert_boxes))

        tta_enabled = any(m.get("name") == "tta" and m.get("enabled") for m in cfg.get("uncertainty", {}).get("methods", []))
        tta_method = next((m for m in cfg.get("uncertainty", {}).get("methods", []) if m.get("name") == "tta" and m.get("enabled")), None)
        tta_images = generate_tta_images(image, tta_method.get("augmentations", []))[: int(tta_method.get("num_samples", 8))] if tta_method else []
        tta_predictions = [detector.predict(ti) for ti in tta_images]

        for j, d in enumerate(dets):
            matched, best_iou, _ = match_detection_to_ground_truth(d.box_xyxy, d.cls, gt_boxes, gt_labels, iou_threshold=iou_threshold, require_class_match=True)
            is_correct = 1 if matched else 0

            tta_confs: List[float] = []
            for tdets in tta_predictions:
                _, _, confs = _match_reference_to_probe(d, tdets, class_match_required=require_class_match)
                if confs:
                    tta_confs.extend(confs)
            tta_std = tta_confidence_std(tta_confs) if tta_confs else 0.0

            matched_boxes: List[np.ndarray] = []
            matched_classes: List[int] = []
            matched_confs: List[float] = []
            for probe_dets, _ in probe_predictions:
                boxes, classes, confs = _match_reference_to_probe(d, probe_dets, class_match_required=require_class_match)
                matched_boxes.extend(boxes)
                matched_classes.extend(classes)
                matched_confs.extend(confs)
            cons = summarize_consistency(d.box_xyxy, matched_boxes, matched_classes, matched_confs, d.cls)

            feats = {
                "confidence": float(d.conf),
                "box_area": float(area(d.box_xyxy)),
                "aspect_ratio": float((d.box_xyxy[2] - d.box_xyxy[0]) / max(1e-6, (d.box_xyxy[3] - d.box_xyxy[1]))),
                "entropy": float(entropy_from_conf(d.conf)),
                "tta_confidence_std": float(tta_std),
                **cons,
            }

            X_rows.append([float(feats.get(k, 0.0)) for k in feature_names])
            y_incorrect.append(1 - is_correct)
            calibration_true.append(is_correct)
            calibration_prob.append(float(d.conf))

    row: Dict[str, Any] = {"run_name": cfg["run"]["name"], "n_images": total_images, "n_detections": total_detections}

    if calibration_prob:
        cal = evaluate_calibration(np.array(calibration_true, dtype=int), np.array(calibration_prob, dtype=float), n_bins=int(cfg["evaluation"]["calibration"].get("ece_bins", 15)))
        row["ece_conf"] = cal.ece
        row["brier_conf"] = cal.brier
        save_reliability_diagram(out_dir, np.array(calibration_true, dtype=int), np.array(calibration_prob, dtype=float), n_bins=int(cfg["evaluation"]["calibration"].get("ece_bins", 15)))
    else:
        row["ece_conf"] = float("nan")
        row["brier_conf"] = float("nan")

    if X_rows and len(set(y_incorrect)) > 1:
        X = np.array(X_rows, dtype=float)
        y_inc = np.array(y_incorrect, dtype=int)
        meta = build_meta_model(cfg["meta_error_predictor"], feature_names)
        indices = np.arange(len(y_inc))
        train_idx, test_idx = train_test_split(indices, test_size=1 - float(cfg["meta_error_predictor"]["training"].get("train_ratio", 0.7)), random_state=seed, stratify=y_inc)
        meta.fit(X[train_idx], y_inc[train_idx])
        p_inc = meta.predict_proba(X[test_idx])
        err = evaluate_error_detection(y_inc[test_idx], p_inc)
        row["meta_error_auroc"] = err.auroc
    else:
        row["meta_error_auroc"] = float("nan")

    if cfg.get("evaluation", {}).get("detection", {}).get("enabled", False):
        det_eval = evaluate_detection_predictions(prediction_records, gt_records)
        row["map50"] = det_eval.map50
        row["map50_95"] = det_eval.map50_95
    else:
        row["map50"] = float("nan")
        row["map50_95"] = float("nan")

    if cfg.get("evaluation", {}).get("ood", {}).get("enabled", False):
        methods = cfg["evaluation"]["ood"].get("scoring", ["max_softmax_prob"])
        for method in methods:
            summary = summarize_ood_scores(ood_detections, method=method)
            row[f"ood_{method}_mean"] = summary.mean_score

    save_results_table(out_dir, row)
    return row


def main():
    ap = argparse.ArgumentParser(prog="sevs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_smoke = sub.add_parser("smoke-test", help="Run end-to-end pipeline on synthetic data")
    p_smoke.add_argument("--config", required=True)
    p_run = sub.add_parser("run", help="Run on dataset from config")
    p_run.add_argument("--config", required=True)

    args = ap.parse_args()
    cfg = load_yaml(args.config)
    logger.info(f"Loaded config: {args.config}")
    row = run_pipeline(cfg)
    logger.info(f"Done. Summary: {row}")
    print(row)


if __name__ == "__main__":
    main()

