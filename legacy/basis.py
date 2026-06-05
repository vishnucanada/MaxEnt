"""Basis-function reconstruction (least-squares) experiment.

The alternative approach from the README: reconstruct each column with an
exponential / polynomial / trigonometric basis via least squares, then
compare actual vs simulated entropy and fidelity. Does not scale (dense
n x n design matrix) but kept for comparison.

Run from the repo root:  python -m legacy.basis
"""

import warnings

import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score

from legacy.io import load_csv, numeric_columns
from legacy.metrics import histogram_probabilities, entropy, compute_fidelity
from legacy.plotting import plot_basis_analysis

warnings.filterwarnings("ignore")

DATASET = "cars_data.csv"
BASIS_NAMES = {"1": "Exponential", "2": "Polynomial", "3": "Trigonometric"}


def generate_basis(x, n, kind="2"):
    if kind == "1":
        return np.exp(-np.outer(x, np.arange(n + 1)))
    if kind == "2":
        return np.power.outer(x, np.arange(n + 1))
    if kind == "3":
        basis = np.ones((len(x), 2 * (n + 1)))
        basis[:, 1] = x
        for i in range(n):
            basis[:, 2 * i + 2] = np.sin(2 * np.pi * (i + 1) ** x)
            basis[:, 2 * i + 3] = np.cos(2 * np.pi * (i + 1) ** x)
        return basis
    raise ValueError(f"unknown basis kind: {kind!r}")


def analysis(data):
    y = np.asarray(data)
    x = np.arange(len(y))
    n = len(y)

    for kind in ("1", "2", "3"):
        try:
            design_matrix = np.vstack(generate_basis(x, n, kind))
            coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
            predictions = design_matrix @ coefficients

            rmse_val = np.sqrt(root_mean_squared_error(predictions, y))
            r2 = r2_score(predictions, y)

            actual = histogram_probabilities(y)
            simulated = histogram_probabilities(predictions)
            entropy_actual = entropy(actual.values())
            entropy_pred = entropy(simulated.values())
            fid = compute_fidelity(predictions, y)

            plot_basis_analysis(x, y, predictions, rmse_val, r2, BASIS_NAMES[kind],
                                entropy_actual, entropy_pred, fid, actual, simulated)
        except (np.linalg.LinAlgError, RuntimeError):
            print("Did not converge")
            continue


def main():
    df = load_csv(DATASET)
    for _, series in numeric_columns(df, head=1000):
        analysis(series)


if __name__ == "__main__":
    main()
