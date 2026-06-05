"""Phase 3 test: accuracy-per-constraint and numerical stability.

Instead of cherry-picking a constraint count, sweep it. For each distribution
we increase the number of scalar moment constraints and record the L1 error of
polynomial-moment maxent vs Fourier-feature (HDC) maxent. The questions:

  1. Which basis reaches low error with FEWER constraints?
  2. Does either basis DESTABILIZE as constraints grow?

Run:  python3.12 -m experiments.hdc_phase3_sweep
"""

import numpy as np

from maxent.hdc import sample_frequencies
from maxent.maxent_fit import fit_maxent, polynomial_features, fourier_features
from maxent.data import OUTPUT_DIR
from experiments.hdc_phase1_kde import sample, true_pdf, l1_error


def run(name, data, grid, truth, n_constraints):
    # polynomial: `n_constraints` moments
    poly_pdf, _, _ = fit_maxent(data, polynomial_features(n_constraints), grid)
    e_poly = l1_error(poly_pdf, truth, grid)

    # fourier: n_constraints//2 frequencies -> n_constraints scalar constraints
    n_freq = max(1, n_constraints // 2)
    bw = 2.0 / (data.std() + 1e-9)
    omega = sample_frequencies(n_freq, bandwidth=bw, kind="gaussian", seed=7)
    four_pdf, _, _ = fit_maxent(data, fourier_features(omega), grid)
    e_four = l1_error(four_pdf, truth, grid)
    return e_poly, e_four


def main():
    rng = np.random.default_rng(0)
    N = 4000
    constraint_counts = [2, 4, 6, 8, 10, 12, 14, 16]

    curves = {}
    for name in ("gaussian", "bimodal", "exponential"):
        data = sample(name, N, rng)
        lo, hi = data.min() - 1.5, data.max() + 1.5
        grid = np.linspace(lo, hi, 800)
        truth = true_pdf(name, grid)

        poly_errs, four_errs = [], []
        for m in constraint_counts:
            ep, ef = run(name, data, grid, truth, m)
            poly_errs.append(ep)
            four_errs.append(ef)
        curves[name] = (poly_errs, four_errs)

        print(f"\n=== {name} ===")
        print(f"{'#constraints':>12}{'poly L1':>10}{'fourier L1':>12}")
        for m, ep, ef in zip(constraint_counts, poly_errs, four_errs):
            print(f"{m:>12}{ep:>10.4f}{ef:>12.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, (poly_errs, four_errs)) in zip(axs, curves.items()):
            ax.plot(constraint_counts, poly_errs, "g-o", label="poly-moment maxent")
            ax.plot(constraint_counts, four_errs, "r-s", label="Fourier (HDC) maxent")
            ax.set_title(name)
            ax.set_xlabel("# scalar constraints")
            ax.set_ylabel("L1 error vs truth")
            ax.set_yscale("log")
            ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "hdc_phase3_sweep.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'hdc_phase3_sweep.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
