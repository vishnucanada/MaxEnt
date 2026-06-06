"""Federated layer: mergeable, DP-composable maxent hypervector sketches.

Each institution turns its local data into one fixed-size sketch (an
unnormalized bundle + a count). Sketches MERGE exactly by count-weighted sum,
compose with the Gaussian mechanism for (epsilon, delta)-differential privacy
(bounded sensitivity because |exp(i w x)| = 1), and the server recovers a
calibrated global density via the maximum-entropy readout.

    per institution :  S_k = sum_x exp(i w x),  N_k = |data_k|     (size M, O(1) memory)
    merge           :  S = sum_k S_k,  N = sum_k N_k
    privatize       :  S += Gaussian noise, sigma ~ sqrt(M) * c(eps,delta) / eps
    readout         :  H = S / N  ->  maxent density over [Re(H), Im(H)]
"""

import numpy as np

from ._compat import trapz
from .hdc import encode
from .maxent_fit import fit_maxent_from_moments, fourier_features


def institution_sketch(data, omega):
    """Local sketch: (unnormalized complex bundle S_k, count N_k). Size M."""
    psi = encode(np.asarray(data, dtype=float), omega)   # (N, M) complex
    psi = np.atleast_2d(psi)
    return psi.sum(axis=0), psi.shape[0]


def merge(sketches):
    """Exact count-weighted merge of [(S_k, N_k), ...] -> (S, N)."""
    S = np.sum([s for s, _ in sketches], axis=0)
    N = int(np.sum([n for _, n in sketches]))
    return S, N


def dp_histogram(banks, grid, edges, epsilon, delta, rng):
    """Baseline: central-DP merged histogram -> density on `grid`.

    Each record contributes to one bin, so the merged histogram has L2
    sensitivity 1; the Gaussian mechanism noises the bin counts.
    """
    counts = np.zeros(len(edges) - 1)
    for d in banks:
        counts += np.histogram(d, bins=edges)[0]
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    counts = np.clip(counts + rng.normal(0, sigma, len(counts)), 0, None)
    centers = (edges[:-1] + edges[1:]) / 2
    area = np.sum(counts * np.diff(edges))
    dens = (counts / area) if area > 0 else counts
    return np.interp(grid, centers, dens)


def gaussian_sigma(epsilon, delta, M):
    """Std-dev per real coordinate for (eps, delta)-DP Gaussian mechanism.

    Add/remove-one sensitivity of the unnormalized sum is ||psi(x)|| = sqrt(M)
    in the 2M-real coordinate space.
    """
    sensitivity = np.sqrt(M)
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def privatize(S, epsilon, delta, rng, local_count=1):
    """Add Gaussian-mechanism noise to a (merged or local) complex sum S.

    `local_count` = number of independently-noised parties contributing to S
    (1 for central DP with secure aggregation; B for local DP where each of B
    institutions noises its own sum). Noise variance adds across parties.
    """
    M = len(S)
    sigma = gaussian_sigma(epsilon, delta, M) * np.sqrt(local_count)
    noise = rng.normal(0, sigma, M) + 1j * rng.normal(0, sigma, M)
    return S + noise


def readout(S, N, omega, grid, reg=0.0):
    """Maximum-entropy global density from a (possibly privatized) sum S.

    Use reg > 0 (relaxed maxent) whenever S carries DP noise -- exact moment
    matching is unstable on noisy/infeasible moments.
    """
    H = S / N
    mu = np.concatenate([H.real, H.imag])
    pdf, _ = fit_maxent_from_moments(mu, fourier_features(omega), grid, reg=reg)
    return pdf


# --- Risk functionals -------------------------------------------------------

def cdf_from_pdf(pdf, grid):
    c = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    return c / c[-1]


def var_es(pdf, grid, level=0.99):
    """Value-at-Risk and Expected Shortfall at `level` (upper tail)."""
    cdf = cdf_from_pdf(pdf, grid)
    var = np.interp(level, cdf, grid)
    mask = grid >= var
    tail_mass = trapz(pdf[mask], grid[mask])
    es = trapz(grid[mask] * pdf[mask], grid[mask]) / tail_mass if tail_mass > 0 else var
    return var, es
