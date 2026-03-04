from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

@dataclass
class MetaErrorModel:
    model: Any
    feature_names: List[str]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # return P(incorrect)=class 1
        proba = self.model.predict_proba(X)
        return proba[:, 1]

def build_meta_model(cfg: Dict[str, Any], feature_names: List[str]) -> MetaErrorModel:
    fam = cfg["model"]["family"]
    params = cfg["model"].get("params", {})
    if fam == "logistic_regression":
        m = LogisticRegression(**params)
        return MetaErrorModel(m, feature_names)
    raise NotImplementedError(f"Meta model '{fam}' not implemented in starter.")
