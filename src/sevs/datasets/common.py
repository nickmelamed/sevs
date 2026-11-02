from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    image_id: str
    gt_boxes_xyxy: np.ndarray
    gt_labels: np.ndarray
    meta: dict[str, Any]


def _load_manifest(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    import json

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def limit_indices(total: int, max_items: int | None = None) -> list[int]:
    if max_items is None or max_items >= total:
        return list(range(total))
    return list(range(max_items))
