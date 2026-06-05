"""maxent: maximum-entropy distribution estimation toolkit.

Public surface re-exported for convenience so experiments can do
`from maxent import fit, pdf, fidelity, ...`.
"""

from .core import pdf, fit, build_constraints, neg_log_likelihood, neg_fidelity
from .metrics import (
    rmse, fidelity, fidelity_from_dicts, entropy,
    histogram_probabilities, compute_fidelity,
)
from .data import (
    load_csv, dataset_path, normalize, numeric_columns,
    DATA_DIR, OUTPUT_DIR, ROOT,
)

__all__ = [
    "pdf", "fit", "build_constraints", "neg_log_likelihood", "neg_fidelity",
    "rmse", "fidelity", "fidelity_from_dicts", "entropy",
    "histogram_probabilities", "compute_fidelity",
    "load_csv", "dataset_path", "normalize", "numeric_columns",
    "DATA_DIR", "OUTPUT_DIR", "ROOT",
]
