"""Method check: bundling data hypervectors reproduces a kernel density estimate.

Confirms the VFA substrate (= Random Fourier Features): the readout of a bundle
matches a Gaussian KDE on gaussian/bimodal; the exponential rings at its hard
boundary, as a smooth kernel must.

Run:  python3.12 -m experiments.method_kde
"""

import numpy as np
from scipy.stats import gaussian_kde

from maxent import sample_frequencies, bundle, density_estimate
from maxent import sample_distribution, true_pdf, l1
from maxent.data import OUTPUT_DIR

DIM, N, BANDWIDTH = 4000, 2000, 2.0


def main():
    rng = np.random.default_rng(0)
    print(f"{'dist':<12}{'HDC L1':>10}{'KDE L1':>10}{'HDC-vs-KDE':>12}")
    print("-" * 44)

    results = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample_distribution(name, N, rng)
        grid = np.linspace(data.min() - 1, data.max() + 1, 600)
        omega = sample_frequencies(DIM, bandwidth=BANDWIDTH, kind="gaussian", seed=1)

        hdc_pdf = density_estimate(bundle(data, omega), grid, omega, normalize_grid=grid)
        kde = gaussian_kde(data)(grid)
        truth = true_pdf(name, grid)

        print(f"{name:<12}{l1(hdc_pdf, truth, grid):>10.4f}{l1(kde, truth, grid):>10.4f}"
              f"{l1(hdc_pdf, kde, grid):>12.4f}")
        results[name] = (grid, truth, kde, hdc_pdf)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, (grid, truth, kde, hdc_pdf)) in zip(axs, results.items()):
            ax.plot(grid, truth, "k-", lw=2, label="true pdf")
            ax.plot(grid, kde, "b--", label="Gaussian KDE")
            ax.plot(grid, hdc_pdf, "r-", alpha=0.8, label="HDC bundle readout")
            ax.set_title(name)
            ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "method_kde.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'method_kde.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
