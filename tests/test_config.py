import re
from pathlib import Path

import pytest

from sevs.utils.config_schema import SCHEMA, unknown_keys, validate_config
from sevs.utils.io import load_yaml

SHIPPED = sorted(Path("configs/eval").glob("*.yaml"))


def leaf_names(schema):
    for key, sub in schema.items():
        yield key
        if isinstance(sub, dict):
            yield from leaf_names(sub)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_configs_use_only_known_keys(path):
    validate_config(load_yaml(str(path)))


def test_unknown_keys_are_named_with_their_path():
    cfg = {"run": {"name": "x", "num_workers": 4}, "reporting": {"save_top_k_failures": 10}}
    assert unknown_keys(cfg) == ["run.num_workers", "reporting.save_top_k_failures"]
    with pytest.raises(ValueError, match="run.num_workers"):
        validate_config(cfg)


def test_every_schema_key_is_read_somewhere_in_the_source():
    source = "\n".join(p.read_text() for p in Path("src/sevs").rglob("*.py") if p.name != "config_schema.py")
    missing = [k for k in set(leaf_names(SCHEMA)) if not re.search(rf"""["']{re.escape(k)}["']""", source)]
    assert missing == []


def base():
    return load_yaml("configs/eval/eval_default.yaml")


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["evaluation"]["detection"].update(metrics=["mAP75"]), "mAP75"),
        (lambda c: c["evaluation"]["ood"].update(scoring=["bogus"]), "bogus"),
        (lambda c: c["uncertainty"]["methods"].append({"name": "mc_dropout", "enabled": True}), "mc_dropout"),
        (lambda c: c["probes"]["perturbations"].append({"name": "x", "severity": 1}), "perturbations"),
        (lambda c: c["probes"]["perturbations"].append({"name": "hflip", "severity": 1, "foo": 2}), "foo"),
        (lambda c: c["reporting"].update(outputs=["ablation_table.md"]), "ablation_table.md"),
        (lambda c: c["detector"].update(family="yolo"), "yolo"),
        (lambda c: c["meta_error_predictor"]["features"].append("colour"), "colour"),
        (lambda c: c["meta_error_predictor"]["model"].update(family="mlp"), "mlp"),
        (lambda c: c["evaluation"]["abstention"].pop("thresholds"), "thresholds"),
        (lambda c: c["uncertainty"]["methods"][0].update(augmentations=["sharpen_2"]), "sharpen"),
    ],
)
def test_bad_values_are_rejected_with_the_offending_text(edit, message):
    cfg = base()
    edit(cfg)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_the_pipeline_validates_its_config(tmp_path):
    from sevs.cli import run_pipeline

    cfg = base()
    cfg["run"]["output_dir"] = str(tmp_path)
    cfg["run"]["num_workers"] = 4
    with pytest.raises(ValueError, match="num_workers"):
        run_pipeline(cfg)


def test_every_schema_key_is_accessed_by_name_in_the_source():
    source = "\n".join(p.read_text() for p in Path("src/sevs").rglob("*.py") if p.name != "config_schema.py")
    missing = [
        k for k in set(leaf_names(SCHEMA)) if not re.search(rf"""(\.get\(|\[)\s*["']{re.escape(k)}["']""", source)
    ]
    assert missing == []
