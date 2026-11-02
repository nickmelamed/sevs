# Changelog

Format: Keep a Changelog. Versioning: Semantic Versioning.

## Unreleased

### Added

- COCO-style mAP with crowd handling, checked against pycocotools.
- Meta-error AUPRC, val and test scores, and abstention coverage and risk.
- Detection post-processing (confidence, class filter, NMS, top-k) and a ROC plot.
- Config key validation and a `results_table.csv` column `is_mock`.
- `pycocotools` as a runtime dependency, and a CI project-checks step.
- `area` and `match_detection_to_ground_truth` in `utils/geometry.py`.
- Ruff, mypy, and pytest settings, and a `dev` extra in `pyproject.toml`.
- Tests for metrics, geometry, mAP, probes, TTA, OOD, and the meta-model.

### Changed

- The meta-model splits images into train, val, and test, and standardizes its features.
- `smoke-test` forces the mock detector and dataset, and `run` refuses them.
- Detections claim ground-truth boxes one-to-one in confidence order.
- Config keys that nothing read were removed, and the rest now act.

### Fixed

- PyTorch is seeded by `seed_everything`.
- TTA detections are mapped back to the original frame.
- The ruff exclude no longer hides `src/sevs`.
- `requirements.txt` no longer installs an old copy of this repo.
- `roc_auc` now counts tied scores as half.
- TTA `scale_*` names changed nothing and `brightness_0.8` brightened the image.
- The CLI failed on import because two geometry helpers were missing.
