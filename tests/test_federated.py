"""Tests for the federated layer (the core invariants behind H1-H3)."""

import numpy as np

from maxent import (
    sample_frequencies, institution_sketch, merge, readout, var_es, l1,
    make_consortium, gaussian_sigma,
)


def _consortium():
    rng = np.random.default_rng(0)
    banks, _ = make_consortium(10, 1500, rng)
    omega = sample_frequencies(32, 1.0, "gaussian", 11)
    return banks, omega


def test_merge_is_lossless():
    """H1: federated merge == centralized estimate on pooled data."""
    banks, omega = _consortium()
    grid = np.linspace(-6, 10, 800)
    S, N = merge([institution_sketch(b, omega) for b in banks])
    S_c, N_c = institution_sketch(np.concatenate(banks), omega)
    assert N == N_c
    assert l1(readout(S, N, omega, grid), readout(S_c, N_c, omega, grid), grid) < 1e-3


def test_sketch_size_constant_in_n():
    """H2: the sketch is fixed-size regardless of local sample count."""
    omega = sample_frequencies(16, 1.0, "gaussian", 0)
    rng = np.random.default_rng(0)
    s_small, _ = institution_sketch(rng.normal(0, 1, 100), omega)
    s_big, _ = institution_sketch(rng.normal(0, 1, 100_000), omega)
    assert s_small.shape == s_big.shape == (16,)


def test_dp_noise_grows_as_epsilon_shrinks():
    assert gaussian_sigma(0.1, 1e-5, 32) > gaussian_sigma(1.0, 1e-5, 32)


def test_var_not_above_es():
    banks, omega = _consortium()
    grid = np.linspace(-6, 10, 800)
    S, N = merge([institution_sketch(b, omega) for b in banks])
    var, es = var_es(readout(S, N, omega, grid), grid, 0.99)
    assert var <= es
