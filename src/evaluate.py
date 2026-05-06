"""Evaluation metrics for price forecasting."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (as a fraction, not %)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that get the sign (direction) correct."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. 1.0 = perfect; 0.0 = matches predicting the mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DirAcc": directional_accuracy(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
    return "  ".join(parts)
