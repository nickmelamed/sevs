from __future__ import annotations
from typing import List
import numpy as np
from .perturbations import apply_perturbation

def generate_tta_images(img: np.ndarray, augmentations: List[str]) -> List[np.ndarray]:
    imgs = [img]
    for aug in augmentations:
        if "_" in aug and aug.split("_")[-1].replace(".","",1).isdigit():
            # e.g., brightness_0.8 handled by mapping to severity approx
            name, val = aug.split("_", 1)
            try:
                f = float(val)
                severity = int(round(abs(f - 1.0) * 10))
            except Exception:
                severity = 1
            imgs.append(apply_perturbation(img, name, max(1, severity)))
        elif aug.startswith("scale_"):
            # lightweight: use random_crop severity proxy
            imgs.append(apply_perturbation(img, "random_crop", 1))
        else:
            imgs.append(apply_perturbation(img, aug, 1))
    return imgs
