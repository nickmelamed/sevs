from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import cv2


@dataclass
class PerturbationResult:
    image: np.ndarray
    invert_boxes: Callable[[np.ndarray], np.ndarray]


def _identity(boxes: np.ndarray) -> np.ndarray:
    return boxes


def apply_perturbation(img: np.ndarray, name: str, severity: int) -> np.ndarray:
    return apply_perturbation_with_inverse(img, name, severity).image


def apply_perturbation_with_inverse(img: np.ndarray, name: str, severity: int) -> PerturbationResult:
    out = img.copy()
    H, W = out.shape[:2]
    if name == "hflip":
        def inv(boxes: np.ndarray) -> np.ndarray:
            b = boxes.copy()
            b[:, 0] = W - boxes[:, 2]
            b[:, 2] = W - boxes[:, 0]
            return b
        return PerturbationResult(image=cv2.flip(out, 1), invert_boxes=inv)
    if name == "gaussian_blur":
        k = 1 + 2 * severity
        return PerturbationResult(image=cv2.GaussianBlur(out, (k, k), 0), invert_boxes=_identity)
    if name == "brightness":
        factor = 1.0 + 0.1 * severity
        return PerturbationResult(image=np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8), invert_boxes=_identity)
    if name == "contrast":
        factor = 1.0 + 0.15 * severity
        mean = out.mean(axis=(0, 1), keepdims=True)
        return PerturbationResult(image=np.clip((out.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8), invert_boxes=_identity)
    if name == "jpeg_compress":
        q = int(max(10, 90 - 20 * severity))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, enc = cv2.imencode(".jpg", out, encode_param)
        return PerturbationResult(image=cv2.imdecode(enc, cv2.IMREAD_COLOR), invert_boxes=_identity)
    if name == "random_crop":
        pad = int(min(H, W) * 0.05 * severity)
        x1 = pad; y1 = pad; x2 = W - pad; y2 = H - pad
        crop = out[y1:y2, x1:x2]
        resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
        sx = (x2 - x1) / max(W, 1)
        sy = (y2 - y1) / max(H, 1)
        def inv(boxes: np.ndarray) -> np.ndarray:
            b = boxes.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sx + x1
            b[:, [1, 3]] = b[:, [1, 3]] * sy + y1
            return b
        return PerturbationResult(image=resized, invert_boxes=inv)
    return PerturbationResult(image=out, invert_boxes=_identity)

