# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and versions follow Semantic Versioning.

## Unreleased

### Added

- Agent tooling (hooks, reviewer agents, rules, skills) and docs/SPEC.md and docs/DECISIONS.md.
- `area` and `match_detection_to_ground_truth` in `utils/geometry.py`.
- Ruff, mypy, and pytest settings, and a `dev` extra in `pyproject.toml`.
- Tests for metrics, geometry, mAP, probes, TTA, OOD, and the meta-model.

### Changed

### Fixed

- `roc_auc` counts tied scores as half. It was wrong with ties.
- TTA `scale_*` names changed nothing and `brightness_0.8` brightened the image.
- The CLI failed on import because two geometry helpers were missing.
