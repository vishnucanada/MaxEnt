"""CSV / dataframe helpers used only by the legacy experiments."""

import numpy as np
import pandas as pd

from maxent.data import DATA_DIR, OUTPUT_DIR  # path constants (one-way dependency)

__all__ = ["DATA_DIR", "OUTPUT_DIR", "load_csv", "normalize", "numeric_columns"]


def load_csv(name):
    return pd.read_csv(DATA_DIR / name).copy()


def normalize(data):
    """Min-max normalize to [0, 1]."""
    data = np.asarray(data, dtype=float)
    return (data - data.min()) / (data.max() - data.min())


def numeric_columns(df, head=None, dropna=True):
    """Yield (column_name, values) for each numeric column."""
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        series = df[col]
        if head is not None:
            series = series.head(head)
        if dropna:
            series = series.dropna()
        yield col, series
