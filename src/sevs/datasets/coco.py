from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from .registry import register

@register("coco")
@dataclass
class CocoDataset:
    cfg: Dict[str, Any]

    def __len__(self) -> int:
        # Starter: use manifest length if present
        manifest = self.cfg.get("subset_manifest")
        if not manifest:
            return 0
        import json
        with open(manifest, "r", encoding="utf-8") as f:
            m = json.load(f)
        return len(m.get("images", []))

    def iter_images(self):
        # Starter placeholder: yield nothing until user wires actual COCO
        return iter([])
