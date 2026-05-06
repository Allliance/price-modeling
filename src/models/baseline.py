"""Naive baselines: Zero (always predict 0) and Naive (persistence on lag-1).

These models do NOT use the scaled feature matrix. The notebook bypasses the
StandardScaler when fitting/predicting these.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


class ZeroBaseline(BaseModel):
    """Always predict zero log-return."""

    name = "Zero"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZeroBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=float)


class NaiveBaseline(BaseModel):
    """Persistence: predict the previous (lag-1) log return.

    Expects the column at ``lag_col_idx`` of X to be the unscaled
    ``log_return_lag1`` feature.
    """

    name = "Naive"

    def __init__(self, lag_col_idx: int = 0):
        self.lag_col_idx = lag_col_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X[:, self.lag_col_idx], dtype=float)


def all_baseline_models(lag_col_idx: int = 0) -> list[BaseModel]:
    return [ZeroBaseline(), NaiveBaseline(lag_col_idx=lag_col_idx)]
