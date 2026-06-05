"""Data loading and the numeric-column iteration shared by every experiment.

Replaces the hardcoded Windows paths
(``C:\\Users\\evgisar\\OneDrive - Ericsson\\...``) that appeared in several
scripts and prevented the project from running anywhere else.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of this package directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def dataset_path(name):
    """Resolve a dataset by filename relative to the bundled data/ dir."""
    return DATA_DIR / name


def load_csv(name):
    return pd.read_csv(dataset_path(name)).copy()


def normalize(data):
    """Min-max normalize to [0, 1]."""
    data = np.asarray(data, dtype=float)
    return (data - data.min()) / (data.max() - data.min())


def numeric_columns(df, head=None, dropna=True):
    """Yield (column_name, values) for each numeric column.

    Replaces the `for col in df.columns: if dtype == object: continue` loop
    that was copy-pasted across the experiment scripts.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        series = df[col]
        if head is not None:
            series = series.head(head)
        if dropna:
            series = series.dropna()
        yield col, series
