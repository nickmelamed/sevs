from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Any, Dict
import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")

def _get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        cur = cur[p]
    return cur

def interpolate(obj: Any, root: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        def repl(m):
            return str(_get_by_path(root, m.group(1)))
        return _VAR.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: interpolate(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate(v, root) for v in obj]
    return obj

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # interpolate after load
    cfg2 = interpolate(cfg, cfg)
    return cfg2

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
