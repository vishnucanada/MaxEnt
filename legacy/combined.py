"""Fit the max-entropy distribution to every numeric column of a dataset,
by both MLE and fidelity, and plot the result.

Run from the repo root:  python -m legacy.combined
"""

import numpy as np

from legacy.io import load_csv, normalize, numeric_columns
from legacy.core import fit
from legacy.plotting import plot_distributions

DATASET = "cars_data.csv"   # or "wind_turbine_data.csv"
INITIAL_GUESS = [0.0, 0.0, 0.0, 0.0]


def main():
    df = load_csv(DATASET)
    subdir = DATASET.replace(".csv", "")

    for col, series in numeric_columns(df, head=2500):
        # Normalization is optional; it just keeps the optimizer well-scaled.
        data = normalize(series.values)
        try:
            lambdas_mle = fit(data, objective="mle", initial_guess=INITIAL_GUESS)
            print(f"Optimal lambdas (MLE) for {col}:", lambdas_mle)

            bins = np.linspace(data.min(), data.max(), 50)
            lambdas_fid = fit(data, objective="fidelity", initial_guess=INITIAL_GUESS, bins=bins)
            print(f"Optimal lambdas (Fidelity) for {col}:", lambdas_fid)

            plot_distributions(data, bins, lambdas_mle, lambdas_fid, col, subdir=subdir)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
        print()


if __name__ == "__main__":
    main()
