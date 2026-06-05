"""Maximum-entropy exponential-family model and fitting.

The model is the generic max-entropy distribution

    p(x) = (1/Z) * exp( sum_i lambda_i * (x - mu)^i )

The lambdas (Lagrange multipliers) are fit either by maximum likelihood
or by maximizing Bhattacharyya fidelity, both subject to the same
normalization + moment constraints. The two optimizers used to be
copy-pasted; they are now a single `fit()` with a pluggable objective.
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import quad

# Slack allowed on each (relaxed) constraint. See the relaxed-moment-constraint
# literature for why a small convex relaxation is preferable to a hard equality.
CONSTRAINT_SLACK = 0.1
LAMBDA_BOUND = 1e5


def pdf(x, lambdas, average):
    """Unnormalized exponential-family density evaluated at x."""
    exponent = np.sum([l * (x - average) ** i for i, l in enumerate(lambdas)], axis=0)
    return np.exp(np.clip(exponent, -700, 700))  # clip to avoid overflow


def normalization_constraint(lambdas, data):
    average = np.mean(data)
    integral, _ = quad(lambda x: pdf(x, lambdas, average), np.min(data), np.max(data))
    return integral - 1


def moment_constraint(lambdas, data, moment_order):
    average = np.mean(data)
    moment_empirical = np.mean((data - average) ** moment_order)
    integral, _ = quad(
        lambda x: (x - average) ** moment_order * pdf(x, lambdas, average),
        np.min(data), np.max(data),
    )
    return integral - moment_empirical


def build_constraints(data, n_lambdas):
    """Normalization + one moment constraint per lambda beyond the first.

    This block was duplicated verbatim in the MLE and fidelity optimizers.
    """
    constraints = [
        NonlinearConstraint(
            lambda lambdas: normalization_constraint(lambdas, data),
            -CONSTRAINT_SLACK, CONSTRAINT_SLACK,
        )
    ]
    for moment_order in range(1, n_lambdas):
        constraints.append(
            NonlinearConstraint(
                lambda lambdas, m=moment_order: moment_constraint(lambdas, data, m),
                -CONSTRAINT_SLACK, CONSTRAINT_SLACK,
            )
        )
    return constraints


# --- Objectives -------------------------------------------------------------

def neg_log_likelihood(lambdas, data):
    if np.any(np.abs(lambdas) > LAMBDA_BOUND):
        return np.inf
    pdf_values = pdf(data, lambdas, np.mean(data))
    log_pdf_values = np.log(pdf_values + 1e-10)  # avoid log(0)
    return -np.sum(log_pdf_values)


def neg_fidelity(lambdas, data, bins):
    from .metrics import fidelity

    average = np.mean(data)
    hist, _ = np.histogram(data, bins=bins, density=True)
    x = (bins[:-1] + bins[1:]) / 2  # bin centers
    predicted = pdf(x, lambdas, average)
    predicted /= np.sum(predicted)
    return -fidelity(hist, predicted)


# --- Unified fit ------------------------------------------------------------

def fit(data, objective="mle", initial_guess=None, bins=None, maxiter=5000):
    """Fit the lambdas. `objective` is "mle" or "fidelity".

    Replaces the former `maximize_log_likelihood` / `maximize_fidelity`,
    which differed only in their objective function.
    """
    if initial_guess is None:
        initial_guess = [0.0, 0.0, 0.0, 0.0]

    if objective == "mle":
        fun, args = neg_log_likelihood, (data,)
    elif objective == "fidelity":
        if bins is None:
            bins = np.linspace(data.min(), data.max(), 50)
        fun, args = neg_fidelity, (data, bins)
    else:
        raise ValueError(f"unknown objective: {objective!r}")

    result = minimize(
        fun, initial_guess, args=args, method="trust-constr",
        constraints=build_constraints(data, len(initial_guess)),
        bounds=[(-LAMBDA_BOUND, LAMBDA_BOUND)] * len(initial_guess),
        options={"verbose": 1, "maxiter": maxiter},
    )
    return result.x
