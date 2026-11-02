from __future__ import annotations

import cv2
import numpy as np

from .perturbations import apply_perturbation


def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _scale(img: np.ndarray, factor: float) -> np.ndarray:
    """Zoom in for factor < 1 and shrink onto a black canvas for factor > 1."""
    h, w = img.shape[:2]
    if factor <= 0 or factor == 1.0:
        return img.copy()
    if factor < 1.0:
        ch, cw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    sh, sw = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros_like(img)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    out[y0:y0 + sh, x0:x0 + sw] = small
    return out


def _split_factor(aug: str) -> tuple[str, float | None]:
    name, _, val = aug.rpartition("_")
    try:
        return name, float(val)
    except ValueError:
        return aug, None


def generate_tta_images(img: np.ndarray, augmentations: list[str]) -> list[np.ndarray]:
    """Return the original image followed by one augmented copy per name.

    A name ending in a number is read as ``<kind>_<factor>``. ``brightness``
    multiplies pixel values by the factor and ``scale`` zooms by it. Other
    kinds map the distance of the factor from 1 to a perturbation severity.
    """
    imgs = [img]
    for aug in augmentations:
        name, factor = _split_factor(aug)
        if factor is None:
            imgs.append(apply_perturbation(img, aug, 1))
        elif name == "brightness":
            imgs.append(_brightness(img, factor))
        elif name == "scale":
            imgs.append(_scale(img, factor))
        else:
            severity = max(1, int(round(abs(factor - 1.0) * 10)))
            imgs.append(apply_perturbation(img, name, severity))
    return imgs
