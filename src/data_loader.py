"""Load and preprocess the raw crypto CSV."""

import pandas as pd


# Columns that are numeric (everything except date and symbol)
_OHLCV = ["open", "high", "low", "close", "volume usdt", "tradecount"]
_INDICATORS = [
    "SMA_10", "SMA_20", "EMA_10", "EMA_20", "EMA_Fast", "EMA_Slow",
    "MACD_Line", "Signal_Line", "MACD_Histogram",
    "BB_Middle", "BB_STD", "BB_Upper", "BB_Lower",
    "Stochastic_High", "Stochastic_Low", "Stochastic_%K", "Stochastic_%D",
    "Previous_Close", "TR_HL", "TR_HC", "TR_LC", "TR", "ATR_14",
    "Parabolic_SAR", "UpMove", "DownMove", "+DM", "-DM",
    "TR1", "TR2", "TR3", "TR_Smoothed", "+DM_Smoothed", "-DM_Smoothed",
    "+DI", "-DI", "DX", "ADX",
    "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B",
    "Chikou_Span", "OBV",
]
NUMERIC_COLS = _OHLCV + _INDICATORS


def load(path: str) -> pd.DataFrame:
    """
    Load the CSV and return a clean DataFrame.

    - `date` parsed as datetime (UTC-naive)
    - All numeric columns cast to float32
    - Sorted by (symbol, date)
    """
    df = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={c: "float32" for c in NUMERIC_COLS},
    )
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
