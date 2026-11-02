# Decisions

Reconstructed from the code. Each entry says what was chosen and the likely reason. Reasons tagged [guess] are inferred.

## D1. Native detection metrics instead of pycocotools
mAP is computed in `evaluation/detection_eval.py` (README: "avoids hard dependence on COCOeval so the repo stays portable"). It reimplements COCOeval's matching and interpolation (D14), so the numbers are comparable with published ones.

## D2. Detector wrappers behind one interface
`BaseDetector.predict(image) -> list[Detection]` with Mock, Torchvision, and Ultralytics implementations. Why [guess]: lets the pipeline run on a laptop with no weights (mock) and swap real detectors by config.

## D3. Mock detector and dataset kept for smoke tests
The README says the mock detector is preserved for smoke tests. Mock output can look like a result, see SPEC rule 3.

## D4. Correctness label is IoU match with class match
A detection is correct if it matches a ground-truth box at `match_iou_threshold` (0.5) with the same class. The label is the target for the meta-model and for calibration.

## D5. Consistency from perturbations, uncertainty from TTA
Perturbation probes feed the five consistency features, and TTA feeds `tta_confidence_std`. Geometric perturbations (hflip, random_crop) map boxes back to the original frame so IoU is comparable. Why [guess]: a detection that moves or changes class under small changes is more likely wrong.

## D6. Each reference detection matched to its single best probe detection
`_match_reference_to_probe` picks the highest-IoU probe box (same class if required). No match contributes nothing, which yields the "worst" consistency values (IoU 0, churn 1).

## D7. Logistic regression as the meta-model
Config exposes `family`, but only `logistic_regression` is implemented. Why [guess]: small, fast, interpretable, and enough for a first result.

## D8. Binary entropy of confidence as the entropy feature
Confidence is treated as the probability of being correct. This is not class-distribution entropy. The OOD "entropy" uses real logits when present and falls back to this.

## D9. OOD score is per-image max over detection scores
`max_softmax_prob` is `1 - conf`, and `energy` uses logits or a two-class pseudo-logit from confidence. Images with no detections score 1.0. Why [guess]: simple, needs no extra model, and works with any detector.

## D10. Config-driven runs, one results row per run
Everything is set in YAML under `configs/`, and a run writes `results_table.csv` plus the reliability diagram to `results/runs/<name>/`, which is gitignored.

## D11. `src/` layout with a local virtualenv named `sevs/`
The package lives in `src/sevs`. The virtualenv sits in the repo root as `sevs/` and is ignored by `/sevs/`. Why [guess]: created by default naming, not a design choice. 

## D12. Labels give each ground-truth box to one detection
`assign_detections_to_ground_truth` visits detections from highest to lowest confidence, and each takes the unmatched ground-truth box with the highest IoU. Duplicate detections of one object are incorrect. This matches how mAP counts them. A detection that stays unmatched but lies on a crowd region of its class (overlap over its own area at least `match_iou_threshold`) is left out of the calibration and meta-model rows, because mAP ignores it too. Labeling always requires a class match. `probes.matching.class_match_required` only affects probe matching. `match_detection_to_ground_truth` still matches one detection on its own.

## D13. The meta-model is split by image
Images are split into train, val, and test with `train_ratio`, `val_ratio`, and `test_ratio`. Detections from one image never appear in two parts. Test AUROC and AUPRC are the headline numbers and val is reported next to them. The split is not stratified by label, because labels belong to detections, not images. The model standardizes features first, so predictions do not depend on feature units.

## D14. mAP follows COCOeval
Matching, the crowd ignore rule, the top-100 cut per class and image, and 101-point interpolation follow pycocotools for the bbox task with the "all" area range. The reason is that results are usually compared with COCO numbers. Precision is tp / (tp + fp) without COCOeval's 1e-16 guard term, so results agree to about 1e-15. Images are ordered by numeric id, as pycocotools does, because tied scores are visited in that order. Parity tests run against pycocotools on random scenes with crowd regions, tied scores, 30 images, and more than 100 detections of a class in one image.

## D15. A config key either does something or is not allowed
`utils/config_schema.py` lists the allowed keys and the allowed values for names such as metrics, perturbations, and features. `run_pipeline` rejects anything else before it starts. A test checks that every allowed key is read by name in the source. Temperature scaling, MC dropout, the ablation table, and the top-k failure dump were removed because nothing implements them. Add them back with the code.

## D16. Mock runs are separate from real runs
`smoke-test` forces the mock detector and dataset and writes to a `*_smoke` directory. `run` refuses a mock config. Every results row has `is_mock`. The reason is SPEC rule 3.

## D17. Post-processing wraps every detector
Confidence threshold, class filter, class-wise NMS, and the top-k cap are applied by one wrapper, so the reference pass, the perturbation probes, and TTA all see the same detections. Torchvision and Ultralytics apply their own NMS before this wrapper.

## D18. TTA names carry a factor
A TTA name ending in a number is read as `<kind>_<factor>`. `brightness_f` multiplies pixel values by f. `scale_f` zooms in for f below 1 and shrinks onto a black canvas for f above 1. Other kinds map abs(f - 1) * 10 to a perturbation severity. Non-finite or non-positive factors raise `ValueError`. Each view carries a function that maps boxes back to the original frame.

## D19. AUROC from average ranks
`roc_auc` computes the Mann-Whitney statistic from average ranks, so tied scores count as half. Single-class input returns NaN.
