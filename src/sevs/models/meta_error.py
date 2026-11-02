from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from sevs.evaluation.error_eval import ErrorDetectionResults, evaluate_error_detection
from sevs.utils.splits import split_image_ids


@dataclass
class MetaErrorModel:
    model: Any
    feature_names: list[str]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # return P(incorrect)=class 1
        proba = self.model.predict_proba(X)
        return proba[:, 1]

def build_meta_model(cfg: dict[str, Any], feature_names: list[str]) -> MetaErrorModel:
    fam = cfg["model"]["family"]
    params = cfg["model"].get("params", {})
    if fam == "logistic_regression":
        m = LogisticRegression(**params)
        return MetaErrorModel(m, feature_names)
    raise NotImplementedError(f"Meta model '{fam}' not implemented in starter.")


@dataclass
class MetaErrorScores:
    val: ErrorDetectionResults
    test: ErrorDetectionResults
    images: dict[str, list[str]]
    y_test: np.ndarray
    p_test: np.ndarray


def fit_and_score_by_image(
    X: np.ndarray,
    y_incorrect: np.ndarray,
    image_ids: list[str],
    cfg: dict[str, Any],
    feature_names: list[str],
    seed: int,
) -> MetaErrorScores:
    """Fit on the train images and score the val and test images.

    Rows are detections and ``image_ids`` gives the image of each row. The
    split is by image, so no image contributes rows to two parts. Scores are
    NaN for a part with a single class, or for all parts if the training rows
    have a single class.
    """
    train_cfg = cfg["training"]
    split = split_image_ids(
        sorted(set(image_ids)),
        float(train_cfg["train_ratio"]),
        float(train_cfg["val_ratio"]),
        float(train_cfg["test_ratio"]),
        seed,
    )
    ids = np.array(image_ids)
    masks = {k: np.isin(ids, v) for k, v in split.items()}
    nan = ErrorDetectionResults(auroc=float("nan"), auprc=float("nan"))
    y = np.asarray(y_incorrect, dtype=int)
    train_y = y[masks["train"]]
    if len(train_y) == 0 or train_y.min() == train_y.max():
        return MetaErrorScores(nan, nan, split, np.zeros(0), np.zeros(0))
    model = build_meta_model(cfg, feature_names)
    model.fit(X[masks["train"]], train_y)
    scored = {}
    for part in ("val", "test"):
        m = masks[part]
        if m.sum() == 0:
            scored[part] = (nan, np.zeros(0), np.zeros(0))
            continue
        p = model.predict_proba(X[m])
        scored[part] = (evaluate_error_detection(y[m], p), y[m], p)
    return MetaErrorScores(scored["val"][0], scored["test"][0], split, scored["test"][1], scored["test"][2])
