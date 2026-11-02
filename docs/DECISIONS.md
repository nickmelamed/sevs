# Decisions

Reconstructed from the code. Each entry says what was chosen and the likely reason. Reasons tagged [guess] are inferred.

## D1. Native detection metrics instead of pycocotools
mAP is computed in `evaluation/detection_eval.py` (README: "avoids hard dependence on COCOeval so the repo stays portable"). `pycocotools` stays as a dependency for future use. Consequence: the AP is VOC-style, so it will not match official COCO numbers.

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

## D12. AUROC from average ranks
`roc_auc` computes the Mann-Whitney statistic from average ranks, so tied scores count as half. The earlier step-wise trapezoid treated ties as ordered and gave wrong values. Single-class input returns NaN.

## D13. TTA names carry a factor
A TTA name ending in a number is read as `<kind>_<factor>`. `brightness_f` multiplies pixel values by f. `scale_f` zooms in for f below 1 and shrinks onto a black canvas for f above 1. Other kinds map abs(f - 1) * 10 to a perturbation severity. Non-finite or non-positive factors raise `ValueError`.

## D14. Detection matching does not consume ground truth
`match_detection_to_ground_truth` matches each detection on its own. Two detections can claim the same ground-truth box, so duplicates all count as correct for calibration and the meta-model. The native mAP does consume boxes.
