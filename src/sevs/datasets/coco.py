from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection

from .common import Sample, _load_manifest, limit_indices
from .registry import register


@register("coco")
@dataclass
class CocoDataset:
    cfg: Dict[str, Any]

    def __post_init__(self) -> None:
        manifest = _load_manifest(self.cfg.get("subset_manifest"))
        self.manifest = manifest or {}
        self.coco_root = Path(
            self.manifest.get("coco_root")
            or self.cfg.get("root")
            or "data/external/coco"
        )
        split = self.manifest.get("split") or self.cfg.get("split") or "val2017"
        if split == "val":
            split = "val2017"
        elif split == "train":
            split = "train2017"
        ann_file = self.cfg.get("ann_file") or (self.coco_root / "annotations" / f"instances_{split}.json")
        img_root = self.cfg.get("img_root") or (self.coco_root / split)

        self.dataset = CocoDetection(root=str(img_root), annFile=str(ann_file))
        self.selected_filenames = set(self.manifest.get("images", [])) if self.manifest.get("images") else None
        self.indices: List[int] = []
        for i in range(len(self.dataset)):
            image_id = self.dataset.ids[i]
            img_info = self.dataset.coco.loadImgs([image_id])[0]
            file_name = img_info["file_name"]
            if self.selected_filenames and file_name not in self.selected_filenames:
                continue
            self.indices.append(i)

        max_items = self.cfg.get("max_items")
        if max_items is not None:
            self.indices = self.indices[: int(max_items)]

    def __len__(self) -> int:
        return len(self.indices)

    def iter_samples(self) -> Iterator[Sample]:
        for idx in self.indices:
            image, anns = self.dataset[idx]
            image_id = self.dataset.ids[idx]
            img_info = self.dataset.coco.loadImgs([image_id])[0]
            image_np = np.array(image.convert("RGB"))
            boxes: List[List[float]] = []
            labels: List[int] = []
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(int(ann["category_id"]))
            yield Sample(
                image=image_np,
                image_id=str(image_id),
                gt_boxes_xyxy=np.array(boxes, dtype=float) if boxes else np.zeros((0, 4), dtype=float),
                gt_labels=np.array(labels, dtype=int) if labels else np.zeros((0,), dtype=int),
                meta={"file_name": img_info.get("file_name"), "height": img_info.get("height"), "width": img_info.get("width")},
            )

