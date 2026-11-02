from __future__ import annotations

from typing import Any

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


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` naming every config key that nothing reads."""
    bad = unknown_keys(cfg)
    if bad:
        raise ValueError("Unknown config keys: " + ", ".join(bad))
