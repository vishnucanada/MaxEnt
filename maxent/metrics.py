"""Shared metrics: fidelity, entropy, rmse, and histogram probabilities.

Previously fidelity was defined three times (`quantum_fidelity`, `fidelity`,
`compute_fidelity`), the histogram->probabilities helper twice, and rmse in
several scripts. This is now the single source of truth.
"""

import math
from collections import Counter

import numpy as np


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


def fidelity(p, q):
    """Bhattacharyya coefficient of two probability arrays."""
    return np.sum(np.sqrt(np.asarray(p) * np.asarray(q)))


def fidelity_from_dicts(phi, psi):
    """Bhattacharyya coefficient over two {bin_label: prob} dicts."""
    keys = phi.keys() & psi.keys()
    return sum(np.sqrt(phi[k] * psi[k]) for k in keys)


def entropy(probabilities):
    """Shannon entropy (bits) of an iterable of probabilities."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def histogram_probabilities(numbers, bin_percentage=0.05, bin_edges=None):
    """Bin `numbers` and return {bin_label: probability}.

    If `bin_edges` is given they are used directly (so two datasets can share
    a common binning); otherwise edges are derived from `numbers`.
    """
    numbers = np.asarray(numbers)
    if bin_edges is None:
        num_bins = math.ceil(bin_percentage * len(numbers))
        bin_edges = np.linspace(numbers.min(), numbers.max(), num_bins + 1)

    bin_indices = np.digitize(numbers, bins=bin_edges, right=True) - 1
    total = len(bin_indices)
    counts = Counter(bin_indices)
    return {
        f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]": counts.get(i, 0) / total
        for i in range(len(bin_edges) - 1)
    }


def compute_fidelity(numbers1, numbers2, bin_percentage=0.05):
    """Fidelity between two raw datasets, binned on a shared grid."""
    num_bins = math.ceil(bin_percentage * max(len(numbers1), len(numbers2)))
    lo = min(np.min(numbers1), np.min(numbers2))
    hi = max(np.max(numbers1), np.max(numbers2))
    edges = np.linspace(lo, hi, num_bins + 1)

    p = histogram_probabilities(numbers1, bin_edges=edges)
    q = histogram_probabilities(numbers2, bin_edges=edges)
    return fidelity_from_dicts(p, q)
