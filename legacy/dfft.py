"""Reconstruct a signal via FFT round-trip and report the reconstruction RMSE.

Run from the repo root:  python -m legacy.dfft
"""

import numpy as np
import matplotlib.pyplot as plt

from legacy.metrics import rmse
from maxent.data import OUTPUT_DIR


def unknown_function(x):
    return np.exp(x)


def compute_fourier(points):
    x = np.array(points)
    X = np.fft.fft(x)
    reconstructed = np.fft.ifft(X).real
    return x, reconstructed


def main():
    x_values = list(range(-100, 100))
    y_values = [unknown_function(x) for x in x_values]
    original, reconstructed = compute_fourier(y_values)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(original)), original, label="Original Function")
    plt.plot(range(len(reconstructed)), reconstructed, label="Reconstructed Function", linestyle="dashed")
    plt.title("Original and Reconstructed Functions")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "dfft.png")
    plt.show()

    print(f"RMSE: {rmse(original, reconstructed)}")


if __name__ == "__main__":
    main()
