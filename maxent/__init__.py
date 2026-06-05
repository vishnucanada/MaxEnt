"""maxent: federated, privacy-preserving maximum-entropy distribution estimation.

Public API. Experiments import from here; they never import from one another.

    from maxent import (
        encode, bundle, sample_frequencies,            # HDC / VFA substrate
        fit_maxent, fit_maxent_from_moments,           # convex-dual maxent
        fourier_features, polynomial_features,
        institution_sketch, merge, privatize, readout, # federated layer
        dp_histogram, var_es,
        l1, ks,                                        # metrics
    )
"""

from .hdc import (
    sample_frequencies, encode, bundle, kernel, readout as kde_readout,
    density_estimate,
)
from .maxent_fit import (
    fit_maxent, fit_maxent_from_moments, fourier_features, polynomial_features,
)
from .federated import (
    institution_sketch, merge, privatize, readout, dp_histogram,
    gaussian_sigma, var_es, cdf_from_pdf,
)
from .metrics import l1, ks
from .datasets import (
    true_pdf, sample_distribution, make_consortium, consortium_true_pdf,
)
from .data import ROOT, DATA_DIR, OUTPUT_DIR, dataset_path

__all__ = [
    "sample_frequencies", "encode", "bundle", "kernel", "kde_readout",
    "density_estimate",
    "fit_maxent", "fit_maxent_from_moments", "fourier_features",
    "polynomial_features",
    "institution_sketch", "merge", "privatize", "readout", "dp_histogram",
    "gaussian_sigma", "var_es", "cdf_from_pdf",
    "l1", "ks",
    "true_pdf", "sample_distribution", "make_consortium", "consortium_true_pdf",
    "ROOT", "DATA_DIR", "OUTPUT_DIR", "dataset_path",
]
