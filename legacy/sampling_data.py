"""Guess a column's distribution family from moments, then overlay a sample
from the guessed family against the empirical KDE.

Run from the repo root:  python -m legacy.sampling_data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from legacy.io import load_csv, numeric_columns

DATASET = "cars_data.csv"


def is_close(a, b, tol=5):
    return abs(a - b) < tol


def guess_family(data):
    mean, median = np.mean(data), np.median(data)
    skewness, kurtosis = stats.skew(data), stats.kurtosis(data)
    print(mean, median, skewness, kurtosis)

    if is_close(mean, median) and is_close(skewness, 0) and is_close(kurtosis, 0):
        return "normal"
    if is_close(mean, median) and is_close(skewness, 0) and kurtosis <= 1:
        return "uniform"
    if (not is_close(mean, median)) and skewness > 0 and kurtosis > 1:
        return "exponential"
    return None


def sample_from(family, data):
    if family == "normal":
        return np.random.normal(np.mean(data), np.std(data), len(data))
    if family == "uniform":
        return np.random.uniform(np.min(data), np.max(data), len(data))
    if family == "exponential":
        return np.random.exponential(np.mean(data), len(data))
    return None


def test_prediction(data, family):
    sample = sample_from(family, data)
    _, ax = plt.subplots()
    sns.kdeplot(data, label="Actual Data", ax=ax)
    sns.kdeplot(sample, label=family, ax=ax)
    ax.legend()
    ax.set_title("Density Plot")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    plt.show()


def main():
    df = load_csv(DATASET)
    for _, series in numeric_columns(df):
        family = guess_family(series)
        if family:
            print(f"{family.capitalize()} Dist")
            test_prediction(series, family)


if __name__ == "__main__":
    main()
