"""Phase 2 test: maximum-entropy density from the hypervector basis vs the
classical polynomial-moment maxent.

Claim under test: random-Fourier-feature constraints (the hypervector basis)
recover distributions that polynomial-moment maxent structurally cannot --
especially bimodal, where a degree-4 exponential family has a single mode.

We give BOTH methods the same number of scalar constraints for fairness.

Run:  python3.12 -m experiments.hdc_phase2_maxent
"""

import numpy as np

from maxent.hdc import sample_frequencies
from maxent.maxent_fit import fit_maxent, polynomial_features, fourier_features
from maxent.data import OUTPUT_DIR
from experiments.hdc_phase1_kde import sample, true_pdf, l1_error


def main():
    rng = np.random.default_rng(0)
    N = 4000
    N_FREQ = 4                 # Fourier freqs -> 8 constraints (cos+sin)
    POLY_ORDER = 8             # 8 polynomial constraints (matched count)

    print(f"{'dist':<12}{'poly-maxent L1':>16}{'fourier-maxent L1':>20}")
    print("-" * 48)

    results = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample(name, N, rng)
        lo, hi = data.min() - 1.5, data.max() + 1.5
        grid = np.linspace(lo, hi, 800)
        truth = true_pdf(name, grid)

        # Classical maxent: polynomial moments
        poly_pdf, _, _ = fit_maxent(data, polynomial_features(POLY_ORDER), grid)

        # HDC maxent: random Fourier features. Bandwidth tied to data scale.
        bw = 1.0 / (data.std() + 1e-9)
        omega = sample_frequencies(N_FREQ, bandwidth=bw * 2.0, kind="gaussian", seed=2)
        four_pdf, _, _ = fit_maxent(data, fourier_features(omega), grid)

        e_poly = l1_error(poly_pdf, truth, grid)
        e_four = l1_error(four_pdf, truth, grid)
        print(f"{name:<12}{e_poly:>16.4f}{e_four:>20.4f}")
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
        fig.savefig(OUTPUT_DIR / "hdc_phase2_maxent.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'hdc_phase2_maxent.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
