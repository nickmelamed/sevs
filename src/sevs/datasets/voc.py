from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from .registry import register

@register("voc")
@dataclass
class VocDataset:
    cfg: Dict[str, Any]
    def __len__(self) -> int:
        return 0
    def iter_images(self):
        return iter([])
