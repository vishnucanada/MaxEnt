"""General maximum-entropy density estimation via the convex dual.

Given feature functions phi_1..phi_M and empirical targets mu_j = (1/N) sum_i
phi_j(x_i), the maximum-entropy density consistent with those moments is

    p(x) = (1/Z(lambda)) exp( sum_j lambda_j phi_j(x) )

and the lambdas are found by minimizing the convex dual

    L(lambda) = log Z(lambda) - lambda . mu,        grad L = E_p[phi] - mu.

This single solver is feature-agnostic: pass polynomial features to get the
classical Jaynes maxent, or random-Fourier features (the hypervector basis) to
get the HDC/VFA version. Integration is by grid quadrature, so it is stable and
needs no nonlinear constraint machinery.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def fit_maxent_from_moments(mu, feature_fn, grid):
    """Fit a maximum-entropy density to target moments `mu` directly.

    This is the streaming-friendly entry point: `mu` can come from an online
    bundle hypervector instead of a batch of raw data.

    mu          : (M,) target moments
    feature_fn  : x (array) -> (len(x), M) feature matrix
    grid        : 1-D integration grid spanning the support

    Returns (pdf_on_grid, lambdas).
    """
    phi_grid = feature_fn(grid)                         # (G, M)
    dx = np.gradient(grid)
    log_w = np.log(dx)                                  # quadrature log-weights

    def objective(lam):
        scores = phi_grid @ lam + log_w                 # (G,)
        logZ = logsumexp(scores)
        p = np.exp(scores - logZ)                       # grid probabilities, sum=1
        grad = phi_grid.T @ p - mu
        return logZ - lam @ mu, grad

    res = minimize(objective, np.zeros(mu.shape[0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000})
    lam = res.x
    scores = phi_grid @ lam
    pdf = np.exp(scores - logsumexp(scores + log_w))    # normalized to integrate to 1
    return pdf, lam


def fit_maxent(data, feature_fn, grid):
    """Fit a maximum-entropy density to the empirical moments of `data`.

    Returns (pdf_on_grid, lambdas, target_moments).
    """
    mu = feature_fn(data).mean(axis=0)                  # (M,) empirical moments
    pdf, lam = fit_maxent_from_moments(mu, feature_fn, grid)
    return pdf, lam, mu


# --- Feature factories ------------------------------------------------------

def polynomial_features(order):
    """Classical maxent basis: [x, x^2, ..., x^order] (centered & scaled)."""
    def fn(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.stack([x ** k for k in range(1, order + 1)], axis=1)
    return fn


def fourier_features(omega):
    """Random-Fourier / hypervector basis: [cos(w x), sin(w x)] for each w."""
    omega = np.asarray(omega, dtype=float)
    def fn(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        ang = np.outer(x, omega)                        # (n, M)
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)
    return fn
