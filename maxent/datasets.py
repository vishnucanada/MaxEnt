"""Synthetic test distributions and federated consortia (shared test fixtures).

Kept in the library (not inside experiment scripts) so every experiment imports
the same generators instead of importing helpers from one another.
"""

import numpy as np

# --- 1-D test distributions -------------------------------------------------

def true_pdf(name, x):
    """Analytic density of a named 1-D test distribution."""
    x = np.asarray(x, dtype=float)
    if name == "gaussian":
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    if name == "bimodal":
        a = np.exp(-0.5 * ((x + 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
        b = np.exp(-0.5 * ((x - 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
        return 0.5 * a + 0.5 * b
    if name == "exponential":
        return np.where(x >= 0, np.exp(-x), 0.0)
    raise ValueError(f"unknown distribution: {name!r}")


def sample_distribution(name, n, rng):
    """Draw `n` samples from a named 1-D test distribution."""
    if name == "gaussian":
        return rng.normal(0, 1, n)
    if name == "bimodal":
        left = rng.random(n) < 0.5
        return np.where(left, rng.normal(-2, 0.6, n), rng.normal(2, 0.6, n))
    if name == "exponential":
        return rng.exponential(1.0, n)
    raise ValueError(f"unknown distribution: {name!r}")


# --- Federated synthetic consortium ----------------------------------------

def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def make_consortium(n_banks, n_per_bank, rng):
    """Non-IID consortium: each bank = main component N(m_b, 0.8) + a small-weight
    rare tail N(5, 1.2). Returns (list_of_bank_arrays, components) where
    `components` lets `consortium_true_pdf` build the exact global density.
    """
    banks, comps = [], []
    for _ in range(n_banks):
        m_b = rng.uniform(-1.5, 1.5)          # bank-specific center (non-IID)
        q_b = rng.uniform(0.01, 0.05)         # rare large-transaction weight
        n_tail = rng.binomial(n_per_bank, q_b)
        data = np.concatenate([
            rng.normal(m_b, 0.8, n_per_bank - n_tail),
            rng.normal(5.0, 1.2, n_tail),
        ])
        banks.append(data)
        comps.append((n_per_bank, m_b, 0.8, q_b, 5.0, 1.2))
    return banks, comps


def consortium_true_pdf(comps, grid):
    """Exact global mixture density for a `make_consortium` output."""
    N = sum(c[0] for c in comps)
    p = np.zeros_like(grid)
    for n, m_b, s_b, q_b, mt, st in comps:
        p += (n / N) * ((1 - q_b) * _gauss(grid, m_b, s_b) + q_b * _gauss(grid, mt, st))
    return p
