from __future__ import annotations
from typing import Any, Dict, Callable
import numpy as np
import cv2

def apply_perturbation(img: np.ndarray, name: str, severity: int) -> np.ndarray:
    out = img.copy()
    if name == "hflip":
        return cv2.flip(out, 1)
    if name == "gaussian_blur":
        k = 1 + 2*severity  # 3,5,7...
        return cv2.GaussianBlur(out, (k, k), 0)
    if name == "brightness":
        factor = 1.0 + 0.1*severity
        return np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if name == "contrast":
        factor = 1.0 + 0.15*severity
        mean = out.mean(axis=(0,1), keepdims=True)
        return np.clip((out.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    if name == "jpeg_compress":
        q = int(max(10, 90 - 20*severity))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, enc = cv2.imencode(".jpg", out, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if name == "random_crop":
        H, W = out.shape[:2]
        pad = int(min(H, W) * 0.05 * severity)
        x1 = pad; y1 = pad; x2 = W - pad; y2 = H - pad
        crop = out[y1:y2, x1:x2]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    return out
