from __future__ import annotations

from typing import Any

PERTURBATIONS = {"hflip", "gaussian_blur", "brightness", "contrast", "jpeg_compress", "random_crop"}
TTA_KINDS = PERTURBATIONS | {"scale"}
FEATURES = {
    "confidence",
    "box_area",
    "aspect_ratio",
    "entropy",
    "tta_confidence_std",
    "iou_stability",
    "center_jitter",
    "size_jitter",
    "class_churn_rate",
    "confidence_variance",
}

# Allowed values for a key, and for each element when the value is a list.
VALUES: dict[str, set[str]] = {
    "dataset.name": {"mock", "coco", "voc"},
    "detector.family": {"mock", "torchvision", "ultralytics"},
    "meta_error_predictor.model.family": {"logistic_regression"},
    "meta_error_predictor.features": FEATURES,
    "evaluation.detection.metrics": {"mAP50", "mAP50_95"},
    "evaluation.calibration.metrics": {"ece", "brier"},
    "evaluation.error_detection.metrics": {"auroc", "auprc"},
    "evaluation.ood.scoring": {"max_softmax_prob", "entropy", "energy"},
    "reporting.outputs": {"results_table.csv", "reliability_diagram.png", "roc_error_detection.png"},
}
ITEM_KEYS = {
    "uncertainty.methods": {"name", "enabled", "num_samples", "augmentations"},
    "probes.perturbations": {"name", "severity"},
}

# A dict lists the keys allowed at that level. None accepts anything below it.
SCHEMA: dict[str, Any] = {
    "run": {"name": None, "seed": None, "output_dir": None, "device": {"backend": None}},
    "dataset": {
        "name": None,
        "split": None,
        "max_items": None,
        "subset_manifest": None,
        "class_filter": None,
        "root": None,
        "year": None,
        "ann_file": None,
        "img_root": None,
    },
    "detector": {
        "family": None,
        "variant": None,
        "weights": None,
        "device": None,
        "conf_threshold": None,
        "nms_iou_threshold": None,
        "max_detections_per_image": None,
    },
    "ground_truth": {"match_iou_threshold": None, "ignore_crowd": None, "min_box_area": None},
    "uncertainty": {"enabled": None, "methods": None},
    "probes": {
        "enabled": None,
        "perturbations": None,
        "matching": {"iou_match_threshold": None, "class_match_required": None},
    },
    "meta_error_predictor": {
        "enabled": None,
        "features": None,
        "model": {"family": None, "params": None},
        "training": {"train_ratio": None, "val_ratio": None, "test_ratio": None},
    },
    "evaluation": {
        "detection": {"enabled": None, "metrics": None},
        "calibration": {"enabled": None, "metrics": None, "ece_bins": None},
        "error_detection": {"enabled": None, "metrics": None},
        "abstention": {"enabled": None, "thresholds": None},
        "ood": {"enabled": None, "scoring": None},
    },
    "reporting": {"enabled": None, "outputs": None},
}


def unknown_keys(cfg: dict[str, Any], schema: dict[str, Any] | None = None, prefix: str = "") -> list[str]:
    """Return the dotted paths in ``cfg`` that the schema does not list."""
    schema = SCHEMA if schema is None else schema
    bad = []
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if key not in schema:
            bad.append(path)
        elif isinstance(schema[key], dict) and isinstance(value, dict):
            bad.extend(unknown_keys(value, schema[key], f"{path}."))
    return bad


def _get(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _aug_kind(aug: str) -> str:
    kind, _, factor = aug.rpartition("_")
    try:
        float(factor)
    except ValueError:
        return aug
    return kind


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` listing every unknown key and every value the code cannot use."""
    errors = [f"unknown key {k}" for k in unknown_keys(cfg)]
    for path, allowed in VALUES.items():
        value = _get(cfg, path)
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            if item not in allowed:
                errors.append(f"{path}: {item!r} is not one of {sorted(allowed)}")
    for path, keys in ITEM_KEYS.items():
        for i, item in enumerate(_get(cfg, path) or []):
            errors.extend(f"unknown key {path}[{i}].{k}" for k in item if k not in keys)
    for i, item in enumerate(_get(cfg, "probes.perturbations") or []):
        if item.get("name") not in PERTURBATIONS:
            errors.append(f"probes.perturbations[{i}]: unknown perturbation {item.get('name')!r}")
    for i, item in enumerate(_get(cfg, "uncertainty.methods") or []):
        if item.get("name") != "tta":
            errors.append(f"uncertainty.methods[{i}]: unknown method {item.get('name')!r}, only 'tta' exists")
        for aug in item.get("augmentations", []):
            if _aug_kind(aug) not in TTA_KINDS:
                errors.append(f"uncertainty.methods[{i}].augmentations: unknown kind in {aug!r}")
    meta = _get(cfg, "meta_error_predictor") or {}
    if meta.get("enabled", True) and not meta.get("features"):
        errors.append("meta_error_predictor.features is required when the meta-model is enabled")
    abstention = _get(cfg, "evaluation.abstention") or {}
    if abstention.get("enabled", False) and not abstention.get("thresholds"):
        errors.append("evaluation.abstention.thresholds is required when abstention is enabled")
    if errors:
        raise ValueError("Invalid config: " + "; ".join(errors))
