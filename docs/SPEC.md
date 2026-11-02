# SEVS spec

Status: DRAFT. Items marked **[confirmed]** came from you. Items marked **[inferred]** come from reading the code and need your correction. Items marked **[gap]** are where the code and the stated intent disagree.

## 1. Purpose

SEVS wraps an object detector and estimates when its detections are wrong. It measures how well the detector's confidence is calibrated, probes how stable each detection is under image perturbations and test-time augmentation (TTA), trains a small meta-model that predicts which detections are incorrect, and scores images for out-of-distribution (OOD) risk. [inferred from README and paper outline, please replace with your own one-paragraph summary]

## 2. Rules that must not be broken

These make a reported result wrong or dishonest if violated. **[confirmed]** unless noted.

1. **No test-set leakage.** The meta-error predictor is trained and evaluated on disjoint data. Any calibration fit uses only its calibration split.
2. **Seeded and reproducible.** Every run is determined by `run.seed`. Same config, same data, same code gives the same numbers.
3. **Mock output is never reported as real.** Results from the mock detector or mock dataset must never appear in a results table, README, or paper number as a real result.
4. **Metrics stay correct.** [inferred] mAP, ECE, Brier, and AUROC must match their standard definitions, and a metric must not be swapped silently.
5. **Config is the source of truth.** [inferred] A run's behavior comes from its YAML config, and a config key that is read must do what its name says.

Please add or edit rules here.

## 3. Pipeline (what the code does)

Entry point: `python -m sevs.cli {smoke-test,run} --config <yaml>` (`src/sevs/cli.py`, `run_pipeline`). [inferred]

For each image:
1. The detector predicts boxes, classes, and confidences.
2. Each configured perturbation (hflip, gaussian_blur, brightness, contrast, jpeg_compress, random_crop) is applied and the detector re-run. Boxes from geometric perturbations are mapped back to the original frame.
3. If TTA is enabled, up to `num_samples` augmented images are generated and the detector re-run.
4. Each reference detection is labeled correct or incorrect against ground truth at `match_iou_threshold` with class match required.
5. Each detection gets a feature vector: confidence, box area, aspect ratio, entropy of the confidence, TTA confidence std, and five consistency features.

After all images:
- Calibration: ECE and Brier of raw confidence versus correctness, plus a reliability diagram.
- Meta-error: logistic regression on the features, predicting incorrectness, scored by AUROC on a held-out part.
- Detection mAP@0.50 and mAP@0.50:0.95 (only if `evaluation.detection.enabled`).
- OOD: mean image-level score per method (only if `evaluation.ood.enabled`).
- One-row `results_table.csv` and the reliability PNG are written to `run.output_dir`.

## 4. Components

| Module | Behavior |
|---|---|
| `models/detector.py` | `Detection` dataclass, `MockDetector`, `TorchvisionDetector`, an Ultralytics wrapper, `build_detector`. |
| `datasets/` | COCO and VOC adapters via torchvision, a registry, and subset manifests (`data/splits/*.json`). |
| `probes/perturbations.py` | Six perturbations, each returning the image and a box-inverse function. |
| `probes/tta.py` | Builds TTA images from augmentation names. |
| `probes/consistency.py` | `iou_stability`, `center_jitter`, `size_jitter`, `class_churn_rate`, `confidence_variance` per reference detection. With no matched probe box: IoU 0, churn 1, all else 0. |
| `models/uncertainty.py` | Binary entropy of confidence, and std of TTA confidences. |
| `models/meta_error.py` | Logistic regression returning P(incorrect). Other families raise `NotImplementedError`. |
| `evaluation/detection_eval.py` | Native VOC-style all-point AP, averaged over classes and IoU thresholds. |
| `evaluation/calibration_eval.py`, `utils/metrics.py` | Equal-width-bin ECE, Brier, trapezoid AUROC. |
| `evaluation/ood_eval.py` | Per-detection OOD scores, reduced to one score per image (max by default). |

## 5. Results the project reports

**[confirmed]** The project reports, or plans to report, these results:

- Detection mAP (0.50 and 0.50:0.95) on a COCO val subset and VOC07 test.
- Calibration (ECE, Brier, reliability diagrams).
- Error prediction (AUROC and AUPRC of the meta-model, abstention coverage versus risk).
- OOD detection (max softmax, entropy, energy).

Only mAP, ECE, Brier, error AUROC, and mean OOD score are computed today. AUPRC, abstention curves, and OOD AUROC are in the config or paper outline but have no code. [inferred]

## 6. Known gaps between intent and code

Found while reading. None are fixed yet. The first is a crash, and the rest bear on your rules.

1. **[gap] CLI does not import.** `cli.py` imports `area` and `match_detection_to_ground_truth` from `utils/geometry.py`. Neither exists, so every command fails. You described this as an unfinished refactor, and the intended behavior of both helpers still needs to be written down. [confirmed it is unfinished] Working assumption, accepted for now and open to change: `area(box)` returns the box area, and `match_detection_to_ground_truth(box, cls, gt_boxes, gt_labels, iou_threshold, require_class_match)` returns `(matched, best_iou, gt_index)`, inferred from how `cli.py` unpacks the result.
2. **[gap] Leakage risk (rule 1).** The meta-model split is by detection, not by image, so detections from one image can sit in both train and test. The keys `val_ratio` and `test_ratio` are read from nothing. Only `train_ratio` is used, giving a two-way split.
3. **[gap] Reproducibility (rule 2).** `seed_everything` seeds `random`, NumPy, and `PYTHONHASHSEED`, not PyTorch.
4. **[gap] Mock labeling (rule 3).** The results row has no field saying the detector or dataset was a mock. The mock dataset has empty ground truth, so every detection is incorrect and meta-error AUROC is NaN.
5. **[gap] mAP is not COCOeval.** `detection_eval.py` uses VOC all-point AP and ignores crowd flags (`ground_truth.ignore_crowd` is not read). Its numbers will not match pycocotools and should not be presented as COCO mAP.
6. **[gap] Config keys that do nothing.** `temperature_scaling`, `mc_dropout`, `abstention`, AUPRC, `roc_error_detection.png`, `ablation_table.md`, `save_top_k_failures`, and the `smoke-test` versus `run` distinction (both run the same code).
7. **[gap] TTA augmentation mapping.** In `probes/tta.py`, names like `scale_0.9` match the numeric branch first and become `apply_perturbation("scale", ...)`, which is a no-op. `brightness_0.8` maps to severity 2, which brightens instead of darkening.
8. **[gap] `roc_auc` ignores ties and does not start the curve at (0, 0).**

## 7. Fast checks

**[confirmed]** pytest, ruff, and mypy. Today pytest passes 5 tests, ruff reports 151 findings, and mypy reports 15 errors. They enter the gate one at a time, once each passes.
