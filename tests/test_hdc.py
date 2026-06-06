"""Tests for the VFA / HDC substrate."""

import numpy as np

from maxent import sample_frequencies, bundle, density_estimate, kernel
from maxent._compat import trapz


def test_kernel_peak_at_zero():
    omega = sample_frequencies(2000, 1.0, "gaussian", 0)
    assert abs(kernel(0.0, omega)[0] - 1.0) < 1e-9


def test_bundle_readout_normalized_and_centered():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 3000)
    omega = sample_frequencies(3000, 2.0, "gaussian", 1)
    grid = np.linspace(-5, 5, 600)
    pdf = density_estimate(bundle(data, omega), grid, omega, normalize_grid=grid)
    assert abs(trapz(pdf, grid) - 1.0) < 1e-2        # integrates to 1
    assert abs(grid[np.argmax(pdf)]) < 0.5           # peak near the true mean
