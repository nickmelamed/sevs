from __future__ import annotations

import argparse
import copy
from typing import Any

import numpy as np

from sevs.datasets.registry import get_dataset
from sevs.evaluation.calibration_eval import evaluate_calibration
from sevs.evaluation.detection_eval import evaluate_detection_predictions
from sevs.evaluation.error_eval import abstention_tradeoff
from sevs.evaluation.ood_eval import summarize_ood_scores
from sevs.evaluation.report import save_reliability_diagram, save_results_table, save_roc_diagram
from sevs.logging import get_logger
from sevs.models.detector import Detection, build_detector
from sevs.models.meta_error import fit_and_score_by_image
from sevs.models.postprocess import PostprocessedDetector
from sevs.models.uncertainty import entropy_from_conf, tta_confidence_std
from sevs.probes.consistency import summarize_consistency
from sevs.probes.perturbations import apply_perturbation_with_inverse
from sevs.probes.tta import generate_tta_views
from sevs.utils.config_schema import validate_config
from sevs.utils.geometry import area, assign_detections_to_ground_truth, box_iou
from sevs.utils.ground_truth import prepare_ground_truth
from sevs.utils.io import load_yaml
from sevs.utils.seed import seed_everything

logger = get_logger()


def _mock_image(seed: int = 0, H: int = 480, W: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def _iter_samples(cfg: dict[str, Any]):
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


def _match_reference_to_probe(
    ref_det: Detection, probe_dets: list[Detection], class_match_required: bool = True, min_iou: float = 0.0
) -> tuple[list[np.ndarray], list[int], list[float]]:
    best_iou = 0.0
    best: Detection | None = None
    for d in probe_dets:
        if class_match_required and d.cls != ref_det.cls:
            continue
        iou = box_iou(ref_det.box_xyxy, d.box_xyxy)
        if iou > best_iou and iou >= min_iou:
            best_iou = iou
            best = d
    if best is None:
        return [], [], []
    return [best.box_xyxy], [best.cls], [best.conf]


def is_mock(cfg: dict[str, Any]) -> bool:
    """Return True if the detector or the dataset is the synthetic mock."""
    return cfg["detector"].get("family") == "mock" or cfg["dataset"].get("name") == "mock"


def smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` that runs on mock data and writes to a separate directory."""
    out = copy.deepcopy(cfg)
    out["dataset"]["name"] = "mock"
    out["detector"]["family"] = "mock"
    out["run"]["name"] = f"{out['run']['name']}_smoke"
    out["run"]["output_dir"] = f"{out['run']['output_dir']}_smoke"
    return out


def predict_tta(detector: Any, image: np.ndarray, method: dict[str, Any]) -> list[list[Detection]]:
    """Run the detector on the TTA views and return detections in the original frame."""
    views = generate_tta_views(image, method.get("augmentations", []))[: int(method.get("num_samples", 8))]
    out = []
    for v in views:
        dets = detector.predict(v.image)
        if dets:
            inv = v.invert_boxes(np.stack([d.box_xyxy for d in dets], axis=0))
            for d, box in zip(dets, inv):
                d.box_xyxy = box
        out.append(dets)
    return out


def _on(section: dict[str, Any], default: bool) -> bool:
    return bool(section.get("enabled", default))


def _listed(section: dict[str, Any], name: str) -> bool:
    metrics = section.get("metrics")
    return metrics is None or name in metrics


def run_pipeline(cfg: dict[str, Any], samples: Any = None, detector: Any = None) -> dict[str, Any]:
    seed = int(cfg["run"]["seed"])
    seed_everything(seed)

    out_dir = cfg["run"]["output_dir"]
    det_cfg = cfg["detector"]
    gt_cfg = cfg["ground_truth"]
    class_filter = cfg["dataset"].get("class_filter")
    probes_cfg = cfg.get("probes", {})
    matching = probes_cfg.get("matching", {})
    eval_cfg = cfg.get("evaluation", {})
    reporting = cfg.get("reporting", {})
    outputs = reporting.get("outputs") or []
    write = _on(reporting, True)

    device = cfg.get("run", {}).get("device", {}).get("backend")
    if detector is None:
        detector = build_detector(det_cfg, seed=seed, device=device)
    max_dets = det_cfg.get("max_detections_per_image")
    detector = PostprocessedDetector(
        detector,
        conf_threshold=det_cfg.get("conf_threshold"),
        nms_iou_threshold=det_cfg.get("nms_iou_threshold"),
        max_detections=None if max_dets is None else int(max_dets),
        allowed_classes=class_filter,
    )
    feature_names = cfg["meta_error_predictor"]["features"]
    iou_threshold = float(gt_cfg.get("match_iou_threshold", 0.5))
    require_class_match = bool(matching.get("class_match_required", True))
    min_probe_iou = float(matching.get("iou_match_threshold", 0.0))

    perturbations = probes_cfg.get("perturbations", []) if _on(probes_cfg, True) else []
    uncertainty = cfg.get("uncertainty", {})
    tta_method = None
    if _on(uncertainty, True):
        tta_method = next((m for m in uncertainty.get("methods", []) if m.get("name") == "tta" and m.get("enabled")), None)

    calibration_true: list[int] = []
    calibration_prob: list[float] = []
    X_rows: list[list[float]] = []
    y_incorrect: list[int] = []
    det_image_ids: list[str] = []
    prediction_records: list[tuple[str, list[Detection]]] = []
    gt_records: list[tuple[str, np.ndarray, np.ndarray]] = []
    crowd_records: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ood_detections: list[list[Detection]] = []

    total_images = 0
    total_detections = 0

    for sample in (samples if samples is not None else _iter_samples(cfg)):
        total_images += 1
        image = sample["image"]
        image_id = sample["image_id"]
        gt = prepare_ground_truth(
            sample["gt_boxes_xyxy"],
            sample["gt_labels"],
            sample.get("meta", {}),
            min_box_area=float(gt_cfg.get("min_box_area", 0)),
            class_filter=class_filter,
            ignore_crowd=bool(gt_cfg.get("ignore_crowd", True)),
        )
        gt_boxes, gt_labels = gt.boxes, gt.labels

        dets = detector.predict(image)
        prediction_records.append((image_id, dets))
        gt_records.append((image_id, gt_boxes, gt_labels))
        if len(gt.crowd_boxes):
            crowd_records[image_id] = (gt.crowd_boxes, gt.crowd_labels)
        ood_detections.append(dets)
        total_detections += len(dets)

        probe_predictions: list[tuple[list[Detection], Any]] = []
        for p in perturbations:
            res = apply_perturbation_with_inverse(image, p["name"], int(p.get("severity", 1)))
            probe_dets = detector.predict(res.image)
            if probe_dets:
                inv_boxes = res.invert_boxes(np.stack([d.box_xyxy for d in probe_dets], axis=0))
                for d, inv_box in zip(probe_dets, inv_boxes):
                    d.box_xyxy = inv_box
            probe_predictions.append((probe_dets, res.invert_boxes))

        tta_predictions = predict_tta(detector, image, tta_method) if tta_method else []

        correct_flags = assign_detections_to_ground_truth(
            np.array([d.box_xyxy for d in dets]).reshape(-1, 4),
            np.array([d.cls for d in dets], dtype=int),
            np.array([d.conf for d in dets], dtype=float),
            gt_boxes,
            gt_labels,
            iou_threshold=iou_threshold,
            require_class_match=True,
        )
        for j, d in enumerate(dets):
            is_correct = 1 if correct_flags[j] else 0

            tta_confs: list[float] = []
            for tdets in tta_predictions:
                _, _, confs = _match_reference_to_probe(d, tdets, require_class_match, min_probe_iou)
                if confs:
                    tta_confs.extend(confs)
            tta_std = tta_confidence_std(tta_confs) if tta_confs else 0.0

            matched_boxes: list[np.ndarray] = []
            matched_classes: list[int] = []
            matched_confs: list[float] = []
            for probe_dets, _ in probe_predictions:
                boxes, classes, confs = _match_reference_to_probe(d, probe_dets, require_class_match, min_probe_iou)
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
            det_image_ids.append(str(image_id))
            calibration_true.append(is_correct)
            calibration_prob.append(float(d.conf))

    nan = float("nan")
    row: dict[str, Any] = {
        "run_name": cfg["run"]["name"],
        "is_mock": is_mock(cfg),
        "n_images": total_images,
        "n_detections": total_detections,
    }

    calibration = eval_cfg.get("calibration", {})
    if _on(calibration, True):
        n_bins = int(calibration.get("ece_bins", 15))
        y_cal = np.array(calibration_true, dtype=int)
        p_cal = np.array(calibration_prob, dtype=float)
        cal = evaluate_calibration(y_cal, p_cal, n_bins=n_bins) if len(p_cal) else None
        if _listed(calibration, "ece"):
            row["ece_conf"] = cal.ece if cal else nan
        if _listed(calibration, "brier"):
            row["brier_conf"] = cal.brier if cal else nan
        if cal and write and "reliability_diagram.png" in outputs:
            save_reliability_diagram(out_dir, y_cal, p_cal, n_bins=n_bins)

    error_cfg = eval_cfg.get("error_detection", {})
    meta_cfg = cfg["meta_error_predictor"]
    if _on(meta_cfg, True) and _on(error_cfg, True):
        scores = None
        if X_rows:
            scores = fit_and_score_by_image(
                np.array(X_rows, dtype=float),
                np.array(y_incorrect, dtype=int),
                det_image_ids,
                meta_cfg,
                feature_names,
                seed,
            )
        for metric, attr in (("auroc", "auroc"), ("auprc", "auprc")):
            if _listed(error_cfg, metric):
                row[f"meta_error_{metric}"] = getattr(scores.test, attr) if scores else nan
                row[f"meta_error_{metric}_val"] = getattr(scores.val, attr) if scores else nan
        if scores:
            row.update({f"n_{k}_images": len(v) for k, v in scores.images.items()})
            abstention = eval_cfg.get("abstention", {})
            if _on(abstention, False) and len(scores.y_test):
                for t, coverage, risk in abstention_tradeoff(scores.y_test, scores.p_test, abstention["thresholds"]):
                    row[f"abstain_coverage_{t}"] = coverage
                    row[f"abstain_risk_{t}"] = risk
            if write and "roc_error_detection.png" in outputs and len(set(scores.y_test.tolist())) > 1:
                save_roc_diagram(out_dir, scores.y_test, scores.p_test)

    detection = eval_cfg.get("detection", {})
    if _on(detection, False):
        det_eval = evaluate_detection_predictions(prediction_records, gt_records, crowd=crowd_records)
        if _listed(detection, "mAP50"):
            row["map50"] = det_eval.map50
        if _listed(detection, "mAP50_95"):
            row["map50_95"] = det_eval.map50_95

    ood = eval_cfg.get("ood", {})
    if _on(ood, False):
        for method in ood.get("scoring", ["max_softmax_prob"]):
            summary = summarize_ood_scores(ood_detections, method=method)
            row[f"ood_{method}_mean"] = summary.mean_score

    if write and "results_table.csv" in outputs:
        save_results_table(out_dir, row)
    return row


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="sevs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_smoke = sub.add_parser("smoke-test", help="Run end-to-end pipeline on synthetic data")
    p_smoke.add_argument("--config", required=True)
    p_run = sub.add_parser("run", help="Run on dataset from config")
    p_run.add_argument("--config", required=True)

    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    validate_config(cfg)
    logger.info(f"Loaded config: {args.config}")
    if args.cmd == "smoke-test":
        cfg = smoke_config(cfg)
    elif is_mock(cfg):
        raise SystemExit("Refusing to run: the config uses the mock detector or dataset. Use smoke-test for mock runs.")
    row = run_pipeline(cfg)
    logger.info(f"Done. Summary: {row}")
    print(row)
    return row


if __name__ == "__main__":
    main()

