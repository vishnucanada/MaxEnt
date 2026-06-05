"""Method check: maxent from the Fourier (hypervector) basis vs polynomial moments.

Both fit a maximum-entropy density from the same number of constraints. The
Fourier basis recovers shapes (multimodal, bounded) that low-order polynomial
moments cannot.

Run:  python3.12 -m experiments.method_fourier_vs_poly
"""

import numpy as np

from maxent import sample_frequencies, fit_maxent, polynomial_features, fourier_features
from maxent import sample_distribution, true_pdf, l1
from maxent.data import OUTPUT_DIR

N, N_FREQ, POLY_ORDER = 4000, 4, 8


def main():
    rng = np.random.default_rng(0)
    print(f"{'dist':<12}{'poly-maxent L1':>16}{'fourier-maxent L1':>20}")
    print("-" * 48)

    results = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample_distribution(name, N, rng)
        grid = np.linspace(data.min() - 1.5, data.max() + 1.5, 800)
        truth = true_pdf(name, grid)

        poly_pdf, _, _ = fit_maxent(data, polynomial_features(POLY_ORDER), grid)
        bw = 2.0 / (data.std() + 1e-9)
        omega = sample_frequencies(N_FREQ, bandwidth=bw, kind="gaussian", seed=2)
        four_pdf, _, _ = fit_maxent(data, fourier_features(omega), grid)

        print(f"{name:<12}{l1(poly_pdf, truth, grid):>16.4f}{l1(four_pdf, truth, grid):>20.4f}")
        results[name] = (grid, truth, poly_pdf, four_pdf)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, (grid, truth, poly_pdf, four_pdf)) in zip(axs, results.items()):
            ax.plot(grid, truth, "k-", lw=2, label="true pdf")
            ax.plot(grid, poly_pdf, "g--", label="poly-moment maxent")
            ax.plot(grid, four_pdf, "r-", alpha=0.85, label="Fourier (HDC) maxent")
            ax.set_title(name)
            ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "method_fourier_vs_poly.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'method_fourier_vs_poly.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
