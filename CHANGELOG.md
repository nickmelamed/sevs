# Changelog

Format: Keep a Changelog. Versioning: Semantic Versioning.

## Unreleased

### Added

- `area` and `match_detection_to_ground_truth` in `utils/geometry.py`.
- Ruff, mypy, and pytest settings, and a `dev` extra in `pyproject.toml`.
- Tests for metrics, geometry, mAP, probes, TTA, OOD, and the meta-model.

### Fixed

- `roc_auc` now counts tied scores as half.
- TTA `scale_*` names changed nothing and `brightness_0.8` brightened the image.
- The CLI failed on import because two geometry helpers were missing.
