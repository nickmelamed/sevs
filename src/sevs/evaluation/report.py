from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sevs.utils.io import ensure_dir


def save_results_table(out_dir: str, row: dict[str, Any]) -> str:
    ensure_dir(out_dir)
    df = pd.DataFrame([row])
    path = os.path.join(out_dir, "results_table.csv")
    df.to_csv(path, index=False)
    return path

def save_roc_diagram(out_dir: str, y_incorrect: np.ndarray, p_incorrect: np.ndarray) -> str:
    from sklearn.metrics import roc_curve

    ensure_dir(out_dir)
    fpr, tpr, _ = roc_curve(y_incorrect, p_incorrect)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    path = os.path.join(out_dir, "roc_error_detection.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_reliability_diagram(out_dir: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> str:
    ensure_dir(out_dir)
    bins = np.linspace(0,1,n_bins+1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        xs.append(float((lo+hi)/2))
        ys.append(float(np.mean(y_true[mask])))
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.plot([0,1],[0,1])
    path = os.path.join(out_dir, "reliability_diagram.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
