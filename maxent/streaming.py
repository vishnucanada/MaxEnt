"""Online / streaming density estimation via an incrementally-updated bundle.

The point of doing maxent in the HDC substrate: the constraint vector is a
*bundle* you can update one sample at a time, with no re-optimization over raw
data. A forgetting factor turns it into an exponentially-weighted moving bundle
that tracks a drifting (non-stationary) distribution -- the regime where batch
estimators fail.

    cumulative:  H_t = (n*H_{t-1} + psi(x_t)) / (n+1)        # all data, equal weight
    forgetting:  H_t = (1-f)*H_{t-1} + f*psi(x_t)            # EWMA, window ~ 1/f
"""

import numpy as np

from .hdc import encode, density_estimate
from .maxent_fit import fit_maxent_from_moments, fourier_features


class OnlineBundle:
    """Incrementally accumulate data into a frequency-domain bundle hypervector.

    `forget=None` keeps a running mean over all samples seen. A float in (0, 1]
    is the EWMA weight on the newest sample (smaller -> longer memory).
    """

    def __init__(self, omega, forget=None):
        self.omega = np.asarray(omega, dtype=float)
        self.forget = forget
        self.H = np.zeros(len(self.omega), dtype=complex)
        self.n = 0

    def update(self, x):
        """Fold in a single sample or a batch (processed in order)."""
        psi = encode(x, self.omega)
        psi = np.atleast_2d(psi)
        for p in psi:
            if self.forget is None:
                self.H = (self.n * self.H + p) / (self.n + 1)
            else:
                self.H = (1 - self.forget) * self.H + self.forget * p
            self.n += 1
        return self

    @property
    def moments(self):
        """Target moment vector [Re(H), Im(H)] for the Fourier maxent fit."""
        return np.concatenate([self.H.real, self.H.imag])

    def density_kde(self, grid):
        """Cheap kernel readout (no optimization) of the current bundle."""
        return density_estimate(self.H, grid, self.omega, normalize_grid=grid)

    def density_maxent(self, grid):
        """Maximum-entropy density consistent with the current bundle."""
        pdf, _ = fit_maxent_from_moments(self.moments, fourier_features(self.omega), grid)
        return pdf
