"""Linear models: Ridge, Lasso, ElasticNet."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from .base import BaseModel


class RidgeModel(BaseModel):
    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = Ridge(alpha=alpha, max_iter=5000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LassoModel(BaseModel):
    name = "Lasso"

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self._model = Lasso(alpha=alpha, max_iter=10000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class ElasticNetModel(BaseModel):
    name = "ElasticNet"

    def __init__(self, alpha: float = 1e-4, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


def all_linear_models() -> list[BaseModel]:
    return [
        RidgeModel(alpha=0.1),
        RidgeModel(alpha=1.0),
        RidgeModel(alpha=10.0),
        LassoModel(alpha=1e-5),
        LassoModel(alpha=1e-4),
        ElasticNetModel(alpha=1e-4, l1_ratio=0.5),
    ]
