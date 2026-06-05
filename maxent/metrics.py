"""Distances between densities evaluated on a shared grid."""

import numpy as np


def l1(p, q, grid):
    """L1 (total-variation x2) distance between two densities on `grid`."""
    return np.trapz(np.abs(np.asarray(p) - np.asarray(q)), grid)


def ks(p, q, grid):
    """Kolmogorov-Smirnov distance: max |CDF_p - CDF_q|."""
    cp = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(grid))])
    cq = np.concatenate([[0.0], np.cumsum((q[1:] + q[:-1]) / 2 * np.diff(grid))])
    cp /= cp[-1]
    cq /= cq[-1]
    return np.max(np.abs(cp - cq))
