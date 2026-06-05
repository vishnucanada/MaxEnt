"""Method check (key result): accuracy-per-constraint and numerical stability.

Sweep the number of scalar moment constraints. Polynomial-moment maxent is
ill-conditioned and DIVERGES past ~4-6 moments; the Fourier (hypervector) basis
keeps improving and stays stable. This is the result that motivates using the
Fourier basis -- and, under DP, the relaxed variant.

Run:  python3.12 -m experiments.method_stability
"""

import numpy as np

from maxent import sample_frequencies, fit_maxent, polynomial_features, fourier_features
from maxent import sample_distribution, true_pdf, l1
from maxent.data import OUTPUT_DIR

N = 4000
CONSTRAINTS = [2, 4, 6, 8, 10, 12, 14, 16]


def _errors(name, data, grid, truth, m):
    poly_pdf, _, _ = fit_maxent(data, polynomial_features(m), grid)
    bw = 2.0 / (data.std() + 1e-9)
    omega = sample_frequencies(max(1, m // 2), bandwidth=bw, kind="gaussian", seed=7)
    four_pdf, _, _ = fit_maxent(data, fourier_features(omega), grid)
    return l1(poly_pdf, truth, grid), l1(four_pdf, truth, grid)


def main():
    rng = np.random.default_rng(0)
    curves = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample_distribution(name, N, rng)
        grid = np.linspace(data.min() - 1.5, data.max() + 1.5, 800)
        truth = true_pdf(name, grid)
        poly_errs, four_errs = zip(*[_errors(name, data, grid, truth, m) for m in CONSTRAINTS])
        curves[name] = (poly_errs, four_errs)
        print(f"\n=== {name} ===")
        print(f"{'#constraints':>12}{'poly L1':>10}{'fourier L1':>12}")
        for m, ep, ef in zip(CONSTRAINTS, poly_errs, four_errs):
            print(f"{m:>12}{ep:>10.4f}{ef:>12.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, (poly_errs, four_errs)) in zip(axs, curves.items()):
            ax.plot(CONSTRAINTS, poly_errs, "g-o", label="poly-moment maxent")
            ax.plot(CONSTRAINTS, four_errs, "r-s", label="Fourier (HDC) maxent")
            ax.set_title(name)
            ax.set_xlabel("# scalar constraints")
            ax.set_ylabel("L1 error vs truth")
            ax.set_yscale("log")
            ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "method_stability.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'method_stability.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
