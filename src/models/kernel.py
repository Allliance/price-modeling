"""Kernel methods: Kernel Ridge Regression and SVR.

Both models use Nystroem kernel approximation so they scale linearly in n
instead of O(n^2) memory / O(n^3) time. With ~45k hourly rows per coin the
exact KernelRidge needs a ~16 GB kernel matrix and crashes the kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from .base import BaseModel


class KernelRidgeModel(BaseModel):
    name = "KernelRidge"

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float | None = None,
        n_components: int = 500,
        random_state: int = 0,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self._model = make_pipeline(
            Nystroem(
                kernel=kernel,
                gamma=gamma,
                n_components=n_components,
                random_state=random_state,
            ),
            Ridge(alpha=alpha),
        )

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
        n_components: int = 500,
        random_state: int = 0,
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        # Nystroem doesn't accept "scale"; map it to None (sklearn default 1/n_features).
        nys_gamma = None if isinstance(gamma, str) else gamma
        self._model = make_pipeline(
            Nystroem(
                kernel=kernel,
                gamma=nys_gamma,
                n_components=n_components,
                random_state=random_state,
            ),
            LinearSVR(C=C, epsilon=epsilon, max_iter=5000, random_state=random_state),
        )

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
