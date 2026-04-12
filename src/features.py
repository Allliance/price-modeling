"""Feature engineering for price forecasting.

Target
------
``log_return_1h`` – log return of the *next* close relative to the current close:
    target_t = log(close_{t+1} / close_t)

This is computed per-symbol so the shift never bleeds across coin boundaries.

Features
--------
- All raw technical indicators provided in the CSV (already computed per-symbol)
- Lag features of log-return and close (configurable, default 1–24 h)
- Time-of-day and day-of-week cyclical encodings
- (Joint mode only) one-hot coin indicator columns

After building features every row with NaN is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "log_return_1h"

# Base feature columns from the raw CSV (present for every row after load)
_BASE_FEATURES = [
    "open", "high", "low", "close", "volume usdt", "tradecount",
    "SMA_10", "SMA_20", "EMA_10", "EMA_20", "EMA_Fast", "EMA_Slow",
    "MACD_Line", "Signal_Line", "MACD_Histogram",
    "BB_Middle", "BB_STD", "BB_Upper", "BB_Lower",
    "Stochastic_%K", "Stochastic_%D",
    "TR", "ATR_14", "Parabolic_SAR",
    "+DI", "-DI", "DX", "ADX",
    "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B",
    "OBV",
]


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-period log-return target, grouped by symbol."""
    df = df.copy()
    df[TARGET] = (
        df.groupby("symbol")["close"]
        .transform(lambda s: np.log(s.shift(-1) / s))
    )
    return df


def _add_lag_features(df: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    """Add lagged log-return and close features per symbol."""
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"log_return_lag{lag}"] = (
            df.groupby("symbol")[TARGET].transform(lambda s, l=lag: s.shift(l))
        )
        df[f"close_lag{lag}"] = (
            df.groupby("symbol")["close"].transform(lambda s, l=lag: s.shift(l))
        )
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour-of-day and day-of-week encodings."""
    df = df.copy()
    hour = df["date"].dt.hour
    dow = df["date"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def build(
    df: pd.DataFrame,
    n_lags: int = 24,
    joint: bool = False,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df:
        Raw dataframe from :func:`src.data_loader.load`.
    n_lags:
        Number of hourly lags to include for close price and log-return.
    joint:
        If True, add one-hot coin indicator columns so models can learn
        cross-coin relationships.

    Returns
    -------
    pd.DataFrame with TARGET column and all feature columns; NaN rows dropped.
    """
    df = _add_target(df)
    df = _add_lag_features(df, n_lags=n_lags)
    df = _add_time_features(df)

    if joint:
        dummies = pd.get_dummies(df["symbol"], prefix="coin", dtype="float32")
        df = pd.concat([df, dummies], axis=1)

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame, joint: bool = False) -> list[str]:
    """Return the list of feature column names present in *df*."""
    exclude = {"date", "symbol", TARGET}
    # Also exclude raw columns that are superseded by lag features or target
    lag_cols = [c for c in df.columns if c.startswith(("log_return_lag", "close_lag"))]
    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    coin_cols = [c for c in df.columns if c.startswith("coin_")] if joint else []
    return [
        c for c in df.columns
        if c not in exclude and (
            c in _BASE_FEATURES or c in lag_cols or c in time_cols or c in coin_cols
        )
    ]


def make_Xy(
    df: pd.DataFrame,
    joint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a built DataFrame."""
    feat_cols = get_feature_cols(df, joint=joint)
    X = df[feat_cols].values.astype(np.float64)
    y = df[TARGET].values.astype(np.float64)
    return X, y


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
