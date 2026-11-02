from __future__ import annotations

_REGISTRY: dict[str, type] = {}

def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def get_dataset(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
