from __future__ import annotations

import numpy as np


def split_image_ids(
    image_ids: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    """Split image ids into disjoint train, val, and test lists.

    Splitting by image keeps every detection from one image in one part. The
    result does not depend on the order of ``image_ids``. Train and val sizes
    are rounded, and test takes the remainder.
    """
    if min(train_ratio, val_ratio, test_ratio) < 0 or not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must be non-negative and sum to 1")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Image ids must be unique")
    ordered = sorted(image_ids)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    n_train = int(round(len(shuffled) * train_ratio))
    n_val = min(int(round(len(shuffled) * val_ratio)), len(shuffled) - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
