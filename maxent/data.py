"""Filesystem paths for the project.

Paths resolve relative to the repository root, so the code runs anywhere.
(CSV/dataframe loading helpers used only by the legacy experiments live in
`legacy/io.py`.)
"""

from pathlib import Path

# Repo root = parent of this package directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def dataset_path(name):
    """Resolve a dataset by filename relative to the bundled data/ dir."""
    return DATA_DIR / name
