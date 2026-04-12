"""Train / test split by a datetime cutoff.

All rows with date < cutoff go to train; rows with date >= cutoff go to test.
The cutoff is intentionally strict so that the most recent data is always in
the test set, preventing any look-ahead leakage.
"""

import pandas as pd

# Default cutoff: last ~10 months of the dataset (dataset ends 2023-10-19)
DEFAULT_CUTOFF = "2023-01-01"


def split(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into (train, test) using a hard datetime cutoff.

    Parameters
    ----------
    df:
        Full dataset; must contain a datetime column named *date_col*.
    cutoff:
        ISO-8601 date/datetime string, e.g. ``"2023-01-01"``.
        Rows strictly before this timestamp go to train.
    date_col:
        Name of the datetime column.

    Returns
    -------
    train, test : pd.DataFrame
        Both retain the original index values from *df*.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df[date_col] < cutoff_ts].copy()
    test = df[df[date_col] >= cutoff_ts].copy()
    return train, test


def split_info(train: pd.DataFrame, test: pd.DataFrame, date_col: str = "date") -> str:
    """Return a human-readable summary of the split."""
    lines = [
        f"Train: {len(train):>8,} rows  "
        f"[{train[date_col].min()} → {train[date_col].max()}]",
        f"Test : {len(test):>8,} rows  "
        f"[{test[date_col].min()} → {test[date_col].max()}]",
    ]
    return "\n".join(lines)
