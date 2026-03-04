from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List
import numpy as np

from sevs.logging import get_logger
from sevs.utils.io import load_yaml, ensure_dir
from sevs.utils.seed import seed_everything
from sevs.models.detector import build_detector
from sevs.models.uncertainty import entropy_from_conf, tta_confidence_std
from sevs.probes.tta import generate_tta_images
from sevs.probes.perturbations import apply_perturbation
from sevs.probes.consistency import summarize_consistency
from sevs.models.meta_error import build_meta_model
from sevs.evaluation.calibration_eval import evaluate_calibration
from sevs.evaluation.error_eval import evaluate_error_detection
from sevs.evaluation.report import save_results_table, save_reliability_diagram

logger = get_logger()

def _mock_image(seed: int = 0, H: int = 480, W: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    return img

def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg["run"]["seed"])
    seed_everything(seed)

    out_dir = cfg["run"]["output_dir"]
    ensure_dir(out_dir)

    detector = build_detector(cfg["detector"], seed=seed)

    # Starter: run on a few synthetic images to validate end-to-end signals
    n_images = 12
    all_rows: List[Dict[str, Any]] = []
    y_true_correct = []
    y_prob_correct = []

    feature_names = cfg["meta_error_predictor"]["features"]
    X_rows = []
    y_incorrect = []

    for i in range(n_images):
        img = _mock_image(seed + i)
        dets = detector.predict(img)

        # Create probe images
        perturb_cfg = cfg["probes"]["perturbations"]
        probe_imgs = [apply_perturbation(img, p["name"], int(p.get("severity", 1))) for p in perturb_cfg]

        # For each detection, compute consistency stats via mock matching (same index)
        for j, d in enumerate(dets):
            # mock: "true correctness" is conf>0.5 (replace with IoU matching vs GT)
            is_correct = 1 if d.conf > 0.5 else 0

            # TTA conf samples (mock): re-run detector on TTA images and take same index conf if exists
            tta_enabled = any(m.get("name") == "tta" and m.get("enabled") for m in cfg["uncertainty"]["methods"])
            tta_confs = []
            if tta_enabled:
                tta_method = [m for m in cfg["uncertainty"]["methods"] if m["name"] == "tta"][0]
                tta_imgs = generate_tta_images(img, tta_method.get("augmentations", []))
                for ti in tta_imgs[: int(tta_method.get("num_samples", 8))]:
                    tdets = detector.predict(ti)
                    if j < len(tdets):
                        tta_confs.append(float(tdets[j].conf))
            tta_std = tta_confidence_std(tta_confs) if tta_confs else 0.0

            # mock probe matching: pretend matched boxes are identical
            boxes = [d.box_xyxy for _ in probe_imgs]
            classes = [d.cls for _ in probe_imgs]
            confs = [d.conf for _ in probe_imgs]
            cons = summarize_consistency(d.box_xyxy, boxes, classes, confs, d.cls)

            # build feature vector
            feats = {
                "confidence": float(d.conf),
                "box_area": float(max(0.0, (d.box_xyxy[2]-d.box_xyxy[0]) * (d.box_xyxy[3]-d.box_xyxy[1]))),
                "aspect_ratio": float((d.box_xyxy[2]-d.box_xyxy[0]) / max(1e-6, (d.box_xyxy[3]-d.box_xyxy[1]))),
                "entropy": float(entropy_from_conf(d.conf)),
                "tta_confidence_std": float(tta_std),
                **cons,
            }

            X_rows.append([float(feats.get(k, 0.0)) for k in feature_names])
            y_incorrect.append(1 - is_correct)

            y_true_correct.append(is_correct)
            y_prob_correct.append(float(d.conf))

            all_rows.append({"img_idx": i, "det_idx": j, "is_correct": is_correct, **feats})

    X = np.array(X_rows, dtype=float)
    y_inc = np.array(y_incorrect, dtype=int)
    y_cor = np.array(y_true_correct, dtype=int)
    p_cor = np.array(y_prob_correct, dtype=float)

    # Train meta error predictor
    meta_cfg = cfg["meta_error_predictor"]
    meta = build_meta_model(meta_cfg, feature_names)

    # simple split
    n = len(y_inc)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr = int(n * float(meta_cfg["training"]["train_ratio"]))
    te = n - tr
    tr_idx, te_idx = idx[:tr], idx[tr:]

    meta.fit(X[tr_idx], y_inc[tr_idx])
    p_inc = meta.predict_proba(X[te_idx])

    # Metrics
    cal = evaluate_calibration(y_cor, p_cor, n_bins=int(cfg["evaluation"]["calibration"]["ece_bins"]))
    err = evaluate_error_detection(y_inc[te_idx], p_inc)

    row = {
        "run_name": cfg["run"]["name"],
        "n_detections": int(n),
        "ece_conf": cal.ece,
        "brier_conf": cal.brier,
        "meta_error_auroc": err.auroc,
    }

    # Save report artifacts
    save_results_table(out_dir, row)
    save_reliability_diagram(out_dir, y_cor, p_cor, n_bins=int(cfg["evaluation"]["calibration"]["ece_bins"]))

    return row

def main():
    ap = argparse.ArgumentParser(prog="sevs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke-test", help="Run end-to-end pipeline on synthetic data")
    p_smoke.add_argument("--config", required=True)

    p_run = sub.add_parser("run", help="Alias for smoke-test in starter (wire real data later)")
    p_run.add_argument("--config", required=True)

    args = ap.parse_args()
    cfg = load_yaml(args.config)
    logger.info(f"Loaded config: {args.config}")
    row = run_pipeline(cfg)
    logger.info(f"Done. Summary: {row}")
    print(row)

if __name__ == "__main__":
    main()
