"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Thin wrapper around a sklearn estimator for price forecasting."""

    name: str = "BaseModel"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.name
