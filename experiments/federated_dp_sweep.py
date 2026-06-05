"""Federated validation under differential privacy (H3, synthetic).

Sweep the privacy budget epsilon and compare, on density L1 and VaR@99% error,
our maxent hypervector sketch against the standard DP-histogram baseline. Under
DP the maxent readout MUST be relaxed (reg > 0) -- exact matching on noisy
moments is unbounded.

Run:  python3.12 -m experiments.federated_dp_sweep
"""

import numpy as np

from maxent import (
    sample_frequencies, institution_sketch, merge, privatize, readout, var_es,
    dp_histogram, l1, make_consortium, consortium_true_pdf,
)
from maxent.data import OUTPUT_DIR

B, N_PER_BANK, M, K = 25, 2000, 32, 50
SEEDS = 12
EPSILONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DELTA = 1e-5
DP_REG = 1e-3   # relaxation under DP (fixed, not tuned per-epsilon)


def main():
    grid = np.linspace(-6, 10, 1000)
    edges = np.linspace(-6, 10, K + 1)
    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)

    hdc_l1 = np.zeros((len(EPSILONS), SEEDS))
    his_l1 = np.zeros((len(EPSILONS), SEEDS))
    hdc_var = np.zeros((len(EPSILONS), SEEDS))
    his_var = np.zeros((len(EPSILONS), SEEDS))

    for s in range(SEEDS):
        rng = np.random.default_rng(100 + s)
        banks, comps = make_consortium(B, N_PER_BANK, rng)
        truth = consortium_true_pdf(comps, grid)
        v_true, _ = var_es(truth, grid, 0.99)
        S, N = merge([institution_sketch(d, omega) for d in banks])

        for i, eps in enumerate(EPSILONS):
            hdc_pdf = readout(privatize(S, eps, DELTA, rng), N, omega, grid, reg=DP_REG)
            hdc_l1[i, s] = l1(hdc_pdf, truth, grid)
            hdc_var[i, s] = abs(var_es(hdc_pdf, grid, 0.99)[0] - v_true)

            his_pdf = dp_histogram(banks, grid, edges, eps, DELTA, rng)
            his_l1[i, s] = l1(his_pdf, truth, grid)
            his_var[i, s] = abs(var_es(his_pdf, grid, 0.99)[0] - v_true)

    print(f"{'eps':>6}{'HDC L1':>10}{'Hist L1':>10}{'HDC VaRerr':>12}{'Hist VaRerr':>13}")
    for i, eps in enumerate(EPSILONS):
        print(f"{eps:>6.1f}{hdc_l1[i].mean():>10.4f}{his_l1[i].mean():>10.4f}"
              f"{hdc_var[i].mean():>12.4f}{his_var[i].mean():>13.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        for ax, hdc, his, ylab in [
            (axs[0], hdc_l1, his_l1, "density L1 error"),
            (axs[1], hdc_var, his_var, "VaR@99% error"),
        ]:
            ax.errorbar(EPSILONS, hdc.mean(1), yerr=hdc.std(1), fmt="r-s", capsize=3,
                        label="HDC maxent sketch")
            ax.errorbar(EPSILONS, his.mean(1), yerr=his.std(1), fmt="g-o", capsize=3,
                        label="DP histogram")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("privacy budget  epsilon  (smaller = more private)")
            ax.set_ylabel(ylab)
            ax.legend()
        axs[0].set_title("Density error vs privacy")
        axs[1].set_title("Tail-risk (VaR) error vs privacy")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "federated_dp_sweep.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'federated_dp_sweep.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
