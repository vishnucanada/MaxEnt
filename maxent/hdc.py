"""Hyperdimensional / Vector Function Architecture (VFA) substrate for
representing probability distributions as hypervectors.

Core idea (Frady, Kleyko & Sommer 2022; equivalently Rahimi & Recht's Random
Fourier Features): encode a scalar x as the complex hypervector

    psi(x)_j = exp(i * omega_j * x),   omega_j ~ p(omega)

Then the similarity between two encodings is a shift-invariant kernel

    K(x, y) = (1/D) Re <psi(x), psi(y)> = (1/D) sum_j cos(omega_j (x - y))
            ~= E_omega[cos(omega (x-y))]  = FourierTransform[p(omega)](x-y)

so the *frequency distribution p(omega) chooses the kernel*:
    omega ~ Normal(0, gamma^2)      -> Gaussian kernel  exp(-gamma^2 d^2 / 2)
    omega ~ Uniform(-a, a)          -> sinc kernel       sinc(a d)

Bundling (superposition) a dataset gives a single hypervector whose readout is
a kernel density estimate:

    H = (1/N) sum_i psi(x_i)
    readout(x) = (1/D) Re <psi(x), H> = (1/N) sum_i K(x, x_i)   = KDE

Bundling is incremental (add one term at a time) -> online / streaming density
estimation, which is the hardware-friendly angle.
"""

import numpy as np


def sample_frequencies(dim, bandwidth=1.0, kind="gaussian", seed=None):
    """Draw the random frequencies omega that define the kernel.

    `bandwidth` is the spread of the frequency distribution: larger bandwidth
    -> narrower kernel -> more localized density estimate.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.normal(0.0, bandwidth, dim)
    if kind == "uniform":
        return rng.uniform(-bandwidth, bandwidth, dim)
    raise ValueError(f"unknown frequency kind: {kind!r}")


def encode(x, omega):
    """Fractional-power encoding. Scalar or array x -> complex hypervector(s).

    Returns shape (dim,) for scalar x, or (len(x), dim) for array x.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    psi = np.exp(1j * np.outer(x, omega))  # (n, dim)
    return psi[0] if psi.shape[0] == 1 else psi


def bundle(data, omega):
    """Bundle (superpose) a dataset into one hypervector: H = mean_i psi(x_i)."""
    psi = encode(data, omega)
    return psi.mean(axis=0)


def kernel(delta, omega):
    """The shift-invariant kernel induced by `omega`, evaluated at offset(s)."""
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    return np.cos(np.outer(delta, omega)).mean(axis=1)


def readout(hypervector, query_x, omega):
    """Read the (unnormalized) density estimate out of a bundle hypervector."""
    psi_q = encode(query_x, omega)              # (n, dim) or (dim,)
    return np.real(psi_q @ np.conj(hypervector))


def density_estimate(hypervector, query_x, omega, normalize_grid=None):
    """Readout normalized to integrate to 1 over `normalize_grid` (or query_x)."""
    g = readout(hypervector, query_x, omega)
    g = np.clip(g, 0, None)  # density is non-negative
    grid = query_x if normalize_grid is None else normalize_grid
    area = np.trapz(np.clip(readout(hypervector, grid, omega), 0, None), grid)
    return g / area if area > 0 else g
