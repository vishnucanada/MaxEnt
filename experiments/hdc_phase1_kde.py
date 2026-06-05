"""Phase 1 test: does bundling data hypervectors reproduce the density?

We sample from known distributions, bundle them in the VFA substrate, read the
density back out, and compare to (a) the true pdf and (b) a textbook Gaussian
KDE. If the substrate works, HDC readout ~= KDE ~= truth.

Run:  python3.12 -m experiments.hdc_phase1_kde
"""

import numpy as np
from scipy.stats import gaussian_kde

from maxent.hdc import sample_frequencies, bundle, density_estimate
from maxent.data import OUTPUT_DIR


def true_pdf(name, x):
    if name == "gaussian":
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    if name == "bimodal":
        a = np.exp(-0.5 * ((x + 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
        b = np.exp(-0.5 * ((x - 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
        return 0.5 * a + 0.5 * b
    if name == "exponential":
        return np.where(x >= 0, np.exp(-x), 0.0)
    raise ValueError(name)


def sample(name, n, rng):
    if name == "gaussian":
        return rng.normal(0, 1, n)
    if name == "bimodal":
        comp = rng.random(n) < 0.5
        return np.where(comp, rng.normal(-2, 0.6, n), rng.normal(2, 0.6, n))
    if name == "exponential":
        return rng.exponential(1.0, n)
    raise ValueError(name)


def l1_error(p, q, grid):
    return np.trapz(np.abs(p - q), grid)


def main():
    rng = np.random.default_rng(0)
    DIM = 4000          # hypervector dimensionality
    N = 2000            # data points
    BANDWIDTH = 2.0     # frequency spread -> kernel width

    print(f"{'dist':<12}{'HDC L1':>10}{'KDE L1':>10}{'HDC-vs-KDE':>12}")
    print("-" * 44)

    results = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample(name, N, rng)
        lo, hi = data.min() - 1, data.max() + 1
        grid = np.linspace(lo, hi, 600)

        omega = sample_frequencies(DIM, bandwidth=BANDWIDTH, kind="gaussian", seed=1)
        H = bundle(data, omega)
        hdc_pdf = density_estimate(H, grid, omega, normalize_grid=grid)

        kde = gaussian_kde(data)(grid)
        truth = true_pdf(name, grid)

        e_hdc = l1_error(hdc_pdf, truth, grid)
        e_kde = l1_error(kde, truth, grid)
        e_hdc_kde = l1_error(hdc_pdf, kde, grid)
        print(f"{name:<12}{e_hdc:>10.4f}{e_kde:>10.4f}{e_hdc_kde:>12.4f}")
        results[name] = (grid, truth, kde, hdc_pdf)

    # Save an overlay plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, (grid, truth, kde, hdc_pdf)) in zip(axs, results.items()):
            ax.plot(grid, truth, "k-", label="true pdf", lw=2)
            ax.plot(grid, kde, "b--", label="Gaussian KDE")
            ax.plot(grid, hdc_pdf, "r-", label="HDC bundle readout", alpha=0.8)
            ax.set_title(name)
            ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "hdc_phase1_kde.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'hdc_phase1_kde.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
