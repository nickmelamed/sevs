from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from torchvision.datasets import VOCDetection

from .common import Sample
from .registry import register


VOC_NAME_TO_ID = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}


@register("voc")
@dataclass
class VocDataset:
    cfg: Dict[str, Any]

    def __post_init__(self) -> None:
        root = self.cfg.get("root") or "data/external/voc"
        year = str(self.cfg.get("year") or "2007")
        image_set = self.cfg.get("split") or "test"
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=False)
        max_items = self.cfg.get("max_items")
        self.indices = list(range(len(self.dataset)))
        if max_items is not None:
            self.indices = self.indices[: int(max_items)]

    def __len__(self) -> int:
        return len(self.indices)

    def iter_samples(self) -> Iterator[Sample]:
        for idx in self.indices:
            image, target = self.dataset[idx]
            image_np = np.array(image.convert("RGB"))
            ann = target["annotation"]
            objects = ann.get("object", [])
            if isinstance(objects, dict):
                objects = [objects]
            boxes: List[List[float]] = []
            labels: List[int] = []
            for obj in objects:
                name = obj["name"].lower().strip()
                if name not in VOC_NAME_TO_ID:
                    continue
                b = obj["bndbox"]
                xmin = float(b["xmin"])
                ymin = float(b["ymin"])
                xmax = float(b["xmax"])
                ymax = float(b["ymax"])
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(VOC_NAME_TO_ID[name])
            yield Sample(
                image=image_np,
                image_id=ann.get("filename", str(idx)),
                gt_boxes_xyxy=np.array(boxes, dtype=float) if boxes else np.zeros((0, 4), dtype=float),
                gt_labels=np.array(labels, dtype=int) if labels else np.zeros((0,), dtype=int),
                meta={"folder": ann.get("folder"), "filename": ann.get("filename")},
            )
