from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Detection:
    box_xyxy: np.ndarray
    cls: int
    conf: float
    logits: np.ndarray | None = None


class BaseDetector:
    def predict(self, image: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class MockDetector(BaseDetector):
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, image: np.ndarray) -> List[Detection]:
        H, W = image.shape[:2]
        n = int(self.rng.integers(1, 6))
        dets: List[Detection] = []
        for _ in range(n):
            x1 = float(self.rng.uniform(0, W * 0.7))
            y1 = float(self.rng.uniform(0, H * 0.7))
            x2 = float(min(W, x1 + self.rng.uniform(W * 0.05, W * 0.3)))
            y2 = float(min(H, y1 + self.rng.uniform(H * 0.05, H * 0.3)))
            cls = int(self.rng.integers(0, 5))
            conf = float(self.rng.uniform(0.05, 0.99))
            dets.append(Detection(np.array([x1, y1, x2, y2], dtype=float), cls, conf, logits=np.array([conf, 1-conf], dtype=float)))
        return dets


class TorchvisionDetector(BaseDetector):
    def __init__(self, variant: str = "fasterrcnn_resnet50_fpn_v2", weights: str = "DEFAULT", device: str = "cpu", score_threshold: float = 0.05):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            retinanet_resnet50_fpn_v2,
            ssdlite320_mobilenet_v3_large,
            FasterRCNN_ResNet50_FPN_V2_Weights,
            RetinaNet_ResNet50_FPN_V2_Weights,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        self.device = torch.device(device)
        self.score_threshold = score_threshold

        if variant == "fasterrcnn_resnet50_fpn_v2":
            w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if str(weights).lower() in {"default", "pretrained"} else None
            self.model = fasterrcnn_resnet50_fpn_v2(weights=w)
        elif variant == "retinanet_resnet50_fpn_v2":
            w = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if str(weights).lower() in {"default", "pretrained"} else None
            self.model = retinanet_resnet50_fpn_v2(weights=w)
        elif variant == "ssdlite320_mobilenet_v3_large":
            w = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if str(weights).lower() in {"default", "pretrained"} else None
            self.model = ssdlite320_mobilenet_v3_large(weights=w)
        else:
            raise ValueError(f"Unsupported torchvision variant: {variant}")

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> List[Detection]:
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.to(self.device)
        out = self.model([x])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        dets: List[Detection] = []
        for box, label, score in zip(boxes, labels, scores):
            if float(score) < self.score_threshold:
                continue
            dets.append(Detection(box_xyxy=box.astype(float), cls=int(label), conf=float(score), logits=None))
        return dets


class UltralyticsYOLODetector(BaseDetector):
    def __init__(self, variant: str = "yolov8n.pt", conf_threshold: float = 0.05, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise ImportError("ultralytics is required for YOLO detector support") from e
        self.model = YOLO(variant)
        self.conf_threshold = conf_threshold
        self.device = device

    def predict(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(source=image, conf=self.conf_threshold, device=self.device, verbose=False)
        dets: List[Detection] = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        for box, c, s in zip(xyxy, cls, conf):
            dets.append(Detection(box_xyxy=box.astype(float), cls=int(c), conf=float(s), logits=None))
        return dets


def _resolve_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_detector(cfg: Dict[str, Any], seed: int = 42, device: str | None = None) -> BaseDetector:
    family = cfg.get("family", "mock")
    resolved = _resolve_device(device or cfg.get("device"))
    if family == "mock":
        return MockDetector(seed=seed)
    if family == "torchvision":
        return TorchvisionDetector(
            variant=cfg.get("variant", "fasterrcnn_resnet50_fpn_v2"),
            weights=cfg.get("weights", "DEFAULT"),
            device=resolved,
            score_threshold=float(cfg.get("conf_threshold", 0.05)),
        )
    if family == "ultralytics":
        return UltralyticsYOLODetector(
            variant=cfg.get("variant", "yolov8n.pt"),
            conf_threshold=float(cfg.get("conf_threshold", 0.05)),
            device=resolved,
        )
    raise NotImplementedError(f"Detector family '{family}' not implemented.")
