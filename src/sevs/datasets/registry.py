from __future__ import annotations
from typing import Dict, Type
from dataclasses import dataclass

_REGISTRY: Dict[str, type] = {}

def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def get_dataset(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
