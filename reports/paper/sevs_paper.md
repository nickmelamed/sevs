# SEVS: Self-Evaluating Vision Systems via Uncertainty Calibration and Consistency Probing

## Abstract
Object detectors often output high confidence scores even when detections are wrong, especially under small distribution shifts or image perturbations. We present SEVS, a self-evaluating perception pipeline that augments a base detector with (i) uncertainty and calibration signals, (ii) perturbation-consistency probes, and (iii) a meta error-prediction model that estimates when detections are likely incorrect. SEVS enables reliability reporting and abstention tradeoffs, improving safety and decision-making in downstream systems.

## 1. Introduction
- Why perception reliability matters
- Confidence ≠ correctness
- Our goal: predict and surface *likely failures*

## 2. Related Work
- Uncertainty estimation (MC Dropout, ensembles)
- Calibration (temperature scaling, ECE/Brier)
- Robustness/consistency probing and TTA
- Failure prediction for perception

## 3. Method
### 3.1 Base Detector
### 3.2 Uncertainty & Calibration
### 3.3 Consistency Probing
### 3.4 Meta Error Predictor

## 4. Experiments
- Datasets (COCO/VOC)
- Baselines + ablations
- Metrics: mAP, ECE/Brier, error AUROC/AUPRC, abstention curves

## 5. Results
- Main table + ablation table
- Reliability diagram
- Error ROC

## 6. Discussion
- Failure modes captured vs missed
- Compute/latency tradeoffs
- Limitations + future work (OOD, video)

## 7. Conclusion

## References
See `references.bib`.
