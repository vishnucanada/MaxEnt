"""Tests for the convex-dual maximum-entropy solver."""

import numpy as np

from maxent import (
    fit_maxent, fit_maxent_from_moments, fourier_features, sample_frequencies,
    true_pdf, l1,
)
from maxent._compat import trapz


def test_recovers_gaussian():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 4000)
    grid = np.linspace(-5, 5, 600)
    omega = sample_frequencies(4, 1.0, "gaussian", 2)
    pdf, _, _ = fit_maxent(data, fourier_features(omega), grid)
    assert l1(pdf, true_pdf("gaussian", grid), grid) < 0.1


def test_relaxation_stabilizes_noisy_moments():
    """Regularized maxent stays bounded/normalized on noisy moments and has
    smaller lambdas than the exact (reg=0) fit -- the property DP relies on."""
    rng = np.random.default_rng(0)
    omega = sample_frequencies(8, 1.0, "gaussian", 2)
    grid = np.linspace(-5, 5, 600)
    ff = fourier_features(omega)
    mu_noisy = ff(rng.normal(0, 1, 4000)).mean(0) + rng.normal(0, 0.05, 16)

    pdf, lam = fit_maxent_from_moments(mu_noisy, ff, grid, reg=1e-2)
    assert np.isfinite(pdf).all() and abs(trapz(pdf, grid) - 1.0) < 1e-2

    _, lam_exact = fit_maxent_from_moments(mu_noisy, ff, grid, reg=0.0)
    assert np.linalg.norm(lam) < np.linalg.norm(lam_exact)
