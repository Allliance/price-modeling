"""Kernel methods: Kernel Ridge Regression and SVR."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from .base import BaseModel


class KernelRidgeModel(BaseModel):
    name = "KernelRidge"

    def __init__(self, alpha: float = 1.0, kernel: str = "rbf", gamma: float | None = None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self._model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class SVRModel(BaseModel):
    name = "SVR"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.01,
        gamma: str | float = "scale",
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self._model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


def all_kernel_models() -> list[BaseModel]:
    return [
        KernelRidgeModel(alpha=1.0, kernel="rbf"),
        KernelRidgeModel(alpha=0.1, kernel="rbf"),
        KernelRidgeModel(alpha=1.0, kernel="polynomial"),
        SVRModel(kernel="rbf", C=1.0, epsilon=0.01),
        SVRModel(kernel="rbf", C=10.0, epsilon=0.001),
    ]
