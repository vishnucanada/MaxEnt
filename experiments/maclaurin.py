"""Numerical Taylor/Maclaurin approximation of a sampled function using finite
differences for the first four derivatives.

Run from the repo root:  python -m experiments.maclaurin
"""

import math

import numpy as np
import matplotlib.pyplot as plt

from maxent import rmse
from maxent.data import OUTPUT_DIR


def unknown_function(x):
    return math.sin(x)


def finite_diff(data, i, h, order):
    """Central finite-difference estimate of the `order`-th derivative at i."""
    if order == 1 and 0 < i < len(data) - 1:
        return (data[i + 1] - data[i - 1]) / (2 * h)
    if order == 2 and 0 < i < len(data) - 1:
        return (data[i + 1] - 2 * data[i] + data[i - 1]) / (h ** 2)
    if order == 3 and 1 < i < len(data) - 2:
        return (data[i + 2] - 2 * data[i + 1] + 2 * data[i - 1] - data[i - 2]) / (2 * h ** 3)
    if order == 4 and 1 < i < len(data) - 2:
        return (-data[i + 2] + 4 * data[i + 1] - 6 * data[i] + 4 * data[i - 1] - data[i - 2]) / (h ** 4)
    return None


def taylor_series(data, x, h=1):
    center = len(data) // 2
    total = 0.0
    for order in range(1, 5):
        d = finite_diff(data, center, h, order)
        if d is not None:
            total += d * x ** order / math.factorial(order)
    return total


def main():
    start, end = -80, 100
    sample_points = [unknown_function(x) for x in range(start, end)]
    xpoints = np.arange(start, end)

    approx = [taylor_series(sample_points, x) for x in range(start, end)]

    plt.plot(xpoints, sample_points, label="Original Function")
    plt.plot(xpoints, approx, label="Taylor Series Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Original Function vs Taylor Series Approximation")
    plt.legend()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "polynomial.png")
    plt.show()

    print(f"RMSE: {rmse(sample_points, approx)}")


if __name__ == "__main__":
    main()
