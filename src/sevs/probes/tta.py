from __future__ import annotations

import math

import cv2
import numpy as np

from .perturbations import PerturbationResult, _identity, apply_perturbation_with_inverse


def _brightness(img: np.ndarray, factor: float) -> PerturbationResult:
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return PerturbationResult(image=out, invert_boxes=_identity)


def _scale(img: np.ndarray, factor: float) -> PerturbationResult:
    """Zoom in for factor < 1 and shrink onto a black canvas for factor > 1."""
    h, w = img.shape[:2]
    if factor == 1.0:
        return PerturbationResult(image=img.copy(), invert_boxes=_identity)
    if factor < 1.0:
        ch, cw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

        def inverse_zoom(boxes: np.ndarray) -> np.ndarray:
            b = boxes.copy()
            b[:, [0, 2]] = boxes[:, [0, 2]] * (cw / w) + x0
            b[:, [1, 3]] = boxes[:, [1, 3]] * (ch / h) + y0
            return b

        return PerturbationResult(image=out, invert_boxes=inverse_zoom)
    sh, sw = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros_like(img)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    out[y0:y0 + sh, x0:x0 + sw] = small

    def inverse_shrink(boxes: np.ndarray) -> np.ndarray:
        b = boxes.copy()
        b[:, [0, 2]] = (boxes[:, [0, 2]] - x0) * (w / sw)
        b[:, [1, 3]] = (boxes[:, [1, 3]] - y0) * (h / sh)
        return b

    return PerturbationResult(image=out, invert_boxes=inverse_shrink)


def _split_factor(aug: str) -> tuple[str, float | None]:
    name, _, val = aug.rpartition("_")
    try:
        factor = float(val)
    except ValueError:
        return aug, None
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"Augmentation factor must be positive and finite: {aug}")
    return name, factor


def generate_tta_views(img: np.ndarray, augmentations: list[str]) -> list[PerturbationResult]:
    """Return the original image followed by one augmented view per name.

    Each view carries a function that maps boxes detected in the view back to
    the original image. Names ending in a number are read as
    ``<kind>_<factor>``. Kinds other than brightness and scale map
    ``abs(factor - 1) * 10`` to a perturbation severity. Non-finite or
    non-positive factors raise ``ValueError``.
    """
    views = [PerturbationResult(image=img, invert_boxes=_identity)]
    for aug in augmentations:
        name, factor = _split_factor(aug)
        if factor is None:
            views.append(apply_perturbation_with_inverse(img, aug, 1))
        elif name == "brightness":
            views.append(_brightness(img, factor))
        elif name == "scale":
            views.append(_scale(img, factor))
        else:
            severity = max(1, int(round(abs(factor - 1.0) * 10)))
            views.append(apply_perturbation_with_inverse(img, name, severity))
    return views


def generate_tta_images(img: np.ndarray, augmentations: list[str]) -> list[np.ndarray]:
    """Return the images of ``generate_tta_views``."""
    return [v.image for v in generate_tta_views(img, augmentations)]
