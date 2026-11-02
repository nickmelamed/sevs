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
