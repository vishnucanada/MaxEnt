"""Shared matplotlib helpers for the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from .core import pdf
from .data import OUTPUT_DIR


def plot_distributions(data, bins, lambdas_mle, lambdas_fidelity, col_name, subdir=""):
    """Empirical histogram vs the MLE- and fidelity-fit densities."""
    average = np.mean(data)
    x = np.linspace(data.min(), data.max(), 1000)

    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.5, label="Empirical Distribution")
    plt.plot(x, pdf(x, lambdas_mle, average), label="MLE Predicted PDF", color="red")
    plt.plot(x, pdf(x, lambdas_fidelity, average), label="Fidelity Predicted PDF", color="green")
    plt.title(f"Distribution for {col_name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()

    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{col_name}.png")
    plt.close()


def plot_pdf_from_dict(probabilities, ax=None, color="#0074D9"):
    if ax is None:
        ax = plt.gca()
    n = len(probabilities)
    x_values = np.linspace(0, n, n, endpoint=False)
    ax.bar(x_values, probabilities.values(), width=1, align="center", alpha=0.7, color=color)
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.grid(True)


def plot_basis_analysis(x, y, predictions, rmse, r2, name,
                        entropy_actual, entropy_predicted, fidelity,
                        pdf_data1=None, pdf_data2=None):
    """Three-panel plot for the least-squares basis reconstruction."""
    plt.style.use("ggplot")
    _, axs = plt.subplots(3, 1, figsize=(8, 9))

    axs[0].plot(x, y, color="#0074D9")
    axs[0].plot(x, predictions, linestyle="dashed", color="#FF851B")
    axs[0].legend()
    axs[0].text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=axs[0].transAxes, verticalalignment="top")
    axs[0].text(0.05, 0.90, f"R2: {r2:.2f}", transform=axs[0].transAxes, verticalalignment="top")
    axs[0].set_title("Original Data vs Reconstructed Function")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    plot_pdf_from_dict(pdf_data1, ax=axs[1])
    axs[1].legend()
    axs[1].text(0.95, 0.95, f"Entropy Actual: {entropy_actual:.5f}", transform=axs[1].transAxes,
                verticalalignment="center_baseline", horizontalalignment="right")
    axs[1].text(0.95, 0.87, f"Entropy Simulated: {entropy_predicted:.5f}", transform=axs[1].transAxes,
                verticalalignment="center_baseline", horizontalalignment="right")
    axs[1].text(0.95, 0.79, f"Fidelity: {fidelity:.5f}", transform=axs[1].transAxes,
                verticalalignment="center_baseline", horizontalalignment="right")
    axs[1].set_title("Probability Density Function 1 (PDF)")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Density")
    axs[1].grid(True)

    plot_pdf_from_dict(pdf_data2, ax=axs[2], color="#FF851B")
    axs[2].legend()
    axs[2].set_title("Probability Density Function 2 (PDF)")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("Density")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
