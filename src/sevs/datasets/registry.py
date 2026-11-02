from __future__ import annotations

import importlib

_REGISTRY: dict[str, type] = {}
_ADAPTER_MODULES = {"coco": "sevs.datasets.coco", "voc": "sevs.datasets.voc"}

def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def get_dataset(name: str):
    if name not in _REGISTRY and name in _ADAPTER_MODULES:
        importlib.import_module(_ADAPTER_MODULES[name])
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
