# SEVS spec

Status: draft. Tags: [confirmed] stated by the owner, [inferred] read from the code, [gap] code and intent disagree.

## 1. Purpose

SEVS wraps an object detector and estimates when its detections are wrong. It measures how well the detector's confidence is calibrated, probes how stable each detection is under image perturbations and test-time augmentation (TTA), trains a small meta-model that predicts which detections are incorrect, and scores images for out-of-distribution (OOD) risk.

## 2. Rules that must not be broken

These make a reported result wrong or dishonest if violated. [confirmed] unless noted.

1. No test-set leakage. The meta-error predictor is trained and evaluated on disjoint data. Any calibration fit uses only its calibration split.
2. Seeded and reproducible. Every run is determined by `run.seed`. Same config, same data, same code gives the same numbers.
3. Mock output is never reported as real. Results from the mock detector or mock dataset must never appear in a results table, README, or paper number as a real result.
4. Metrics stay correct. [inferred] mAP, ECE, Brier, and AUROC must match their standard definitions, and a metric must not be swapped silently.
5. Config is the source of truth. [inferred] A run's behavior comes from its YAML config, and a config key that is read must do what its name says.

## 3. Pipeline (what the code does)

Entry point: `python -m sevs.cli {smoke-test,run} --config <yaml>` (`src/sevs/cli.py`, `run_pipeline`). `smoke-test` forces the mock detector and dataset and writes to a `*_smoke` directory. `run` refuses a config that uses either. Unknown config keys are rejected. [inferred]

For each image:
1. The detector predicts boxes, classes, and confidences. Predictions are filtered by `conf_threshold` and `dataset.class_filter`, passed through class-wise NMS at `nms_iou_threshold`, and capped at `max_detections_per_image`.
2. Each configured perturbation (hflip, gaussian_blur, brightness, contrast, jpeg_compress, random_crop) is applied and the detector re-run. Boxes from geometric perturbations are mapped back to the original frame.
3. If TTA is enabled, up to `num_samples` augmented views are generated, the detector is re-run, and boxes are mapped back to the original frame.
4. Detections are labeled correct or incorrect against ground truth at `match_iou_threshold` with class match required. Detections are visited in confidence order and each ground-truth box is used once.
5. Each detection gets a feature vector: confidence, box area, aspect ratio, entropy of the confidence, TTA confidence std, and five consistency features.

After all images:
- Calibration: ECE and Brier of raw confidence versus correctness, plus a reliability diagram. Nothing is fitted, so there is no calibration split.
- Meta-error: images are split into train, val, and test with the configured ratios. A standardized logistic regression is fitted on the train images and scored by AUROC and AUPRC on the test images (headline) and the val images. Abstention coverage and risk are computed on the test images.
- Detection mAP@0.50 and mAP@0.50:0.95 in the style of COCOeval (only if `evaluation.detection.enabled`).
- OOD: mean image-level score per method (only if `evaluation.ood.enabled`).
- The files listed in `reporting.outputs` are written to `run.output_dir`. Every results row has `is_mock`.

## 4. Components

| Module | Behavior |
|---|---|
| `models/detector.py` | `Detection` dataclass, `MockDetector`, `TorchvisionDetector`, an Ultralytics wrapper, `build_detector`. |
| `datasets/` | COCO and VOC adapters via torchvision, a registry, and subset manifests (`data/splits/*.json`). |
| `probes/perturbations.py` | Six perturbations, each returning the image and a box-inverse function. |
| `probes/tta.py` | Builds TTA views from augmentation names, each with a box mapping back to the original frame. |
| `models/postprocess.py` | Confidence and class filter, class-wise NMS, and top-k cap, applied to every detector. |
| `utils/config_schema.py` | The allowed config keys. Unknown keys raise `ValueError`. |
| `utils/splits.py` | Disjoint train, val, and test split of image ids. |
| `probes/consistency.py` | `iou_stability`, `center_jitter`, `size_jitter`, `class_churn_rate`, `confidence_variance` per reference detection. With no matched probe box: IoU 0, churn 1, all else 0. |
| `models/uncertainty.py` | Binary entropy of confidence, and std of TTA confidences. |
| `models/meta_error.py` | Standardized logistic regression returning P(incorrect), fitted and scored by image split. Other families raise `NotImplementedError`. |
| `evaluation/detection_eval.py` | COCOeval-style AP: 101-point interpolation, crowd regions ignored, top 100 detections per class and image. Checked against pycocotools. |
| `evaluation/error_eval.py` | AUROC, AUPRC, and the abstention coverage and risk table. |
| `evaluation/calibration_eval.py`, `utils/metrics.py` | Equal-width-bin ECE, Brier, and rank-based AUROC with ties counted as half. |
| `evaluation/ood_eval.py` | Per-detection OOD scores, reduced to one score per image (max by default). |

## 5. Results the project reports

The project reports, or plans to report [confirmed]:

- Detection mAP (0.50 and 0.50:0.95) on a COCO val subset and VOC07 test.
- Calibration (ECE, Brier, reliability diagrams).
- Error prediction (AUROC and AUPRC of the meta-model, abstention coverage versus risk).
- OOD detection (max softmax, entropy, energy).

mAP, ECE, Brier, error AUROC and AUPRC, abstention coverage and risk, and mean OOD score are computed. OOD AUROC is in the paper outline but has no code, because no labeled out-of-distribution data is wired in. [inferred]

## 6. Known gaps between intent and code

Found while reading the code. Items 1 to 8 are fixed on `main` or in the open pull request. Items 9 to 11 are open.

1. [gap, fixed] CLI did not import. `cli.py` imported `area` and `match_detection_to_ground_truth` from `utils/geometry.py`, and neither existed. Both were lost in an unfinished refactor [confirmed]. Assumed behavior: `area(box)` returns the box area, and `match_detection_to_ground_truth(box, cls, gt_boxes, gt_labels, iou_threshold, require_class_match)` returns `(matched, best_iou, gt_index)`, as `cli.py` unpacked it. The pipeline now labels whole images with `assign_detections_to_ground_truth`.
2. [gap, fixed] Leakage risk (rule 1). The meta-model split was by detection. It is now by image into train, val, and test.
3. [gap, fixed] Reproducibility (rule 2). `seed_everything` did not seed PyTorch.
4. [gap, fixed] Mock labeling (rule 3). `smoke-test` forces mock and `run` refuses it. Every results row has `is_mock`.
5. [gap, fixed] mAP was VOC-style and ignored crowd flags. It now follows COCOeval and is tested against pycocotools.
6. [gap, fixed] Config keys that did nothing. Keys that are implemented now act, and the rest were removed. Unknown keys are rejected.
7. [gap, fixed] TTA augmentation mapping. `scale_*` was a no-op, `brightness_0.8` brightened, and TTA boxes were not mapped back.
8. [gap, fixed] `roc_auc` ignored ties.
9. [gap] The files in `configs/dataset`, `configs/model`, and `configs/probes` are not loaded by anything.
10. [gap] The TTA view set always includes the unmodified image first, so `num_samples` counts it.
11. [gap] No code computes OOD AUROC, because the repo has no labeled OOD data.

## 7. Fast checks

pytest, ruff, and mypy [confirmed]. All three run in `make agent-check`, which the Stop hook uses.
