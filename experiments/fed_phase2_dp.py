"""Federated Phase 2: does the risk tail survive differential privacy?

Headline test (H3). Under central (eps, delta)-DP with secure aggregation, we
compare our maxent hypervector sketch against the standard federated baseline, a
DP histogram, on:
  * density L1 error vs the true global
  * VaR@99% error  (the risk deliverable)
across a sweep of privacy budgets epsilon, averaged over seeds.

Expectation: a histogram must spend its budget across many bins, so noise swamps
the low-count TAIL bins -> VaR blows up at small eps. The maxent sketch noises a
few low-sensitivity moments and smooths -> the tail survives.

Run:  python3.12 -m experiments.fed_phase2_dp
"""

import numpy as np

from maxent.hdc import sample_frequencies
from maxent.federated import (
    institution_sketch, merge, privatize, readout, var_es, gaussian_sigma,
)
from maxent.data import OUTPUT_DIR
from experiments.fed_phase1_synth import make_consortium, true_global_pdf, l1


def dp_histogram(banks, grid, edges, epsilon, delta, rng):
    """Central-DP merged histogram -> density on `grid`."""
    counts = np.zeros(len(edges) - 1)
    for d in banks:
        counts += np.histogram(d, bins=edges)[0]
    # central model: merged histogram, one record -> L2 sensitivity 1
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    counts = counts + rng.normal(0, sigma, len(counts))
    counts = np.clip(counts, 0, None)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    area = np.sum(counts * widths)
    dens = (counts / area) if area > 0 else counts
    return np.interp(grid, centers, dens)


def main():
    B, N_PER_BANK, M, K = 25, 2000, 32, 50
    SEEDS = 12
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    delta = 1e-5
    grid = np.linspace(-6, 10, 1000)
    edges = np.linspace(-6, 10, K + 1)
    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)

    hdc_l1 = np.zeros((len(epsilons), SEEDS))
    his_l1 = np.zeros((len(epsilons), SEEDS))
    hdc_var = np.zeros((len(epsilons), SEEDS))
    his_var = np.zeros((len(epsilons), SEEDS))

    for s in range(SEEDS):
        rng = np.random.default_rng(100 + s)
        banks, comps = make_consortium(B, N_PER_BANK, rng)
        truth = true_global_pdf(comps, grid)
        v_true, _ = var_es(truth, grid, 0.99)

        S, N = merge([institution_sketch(d, omega) for d in banks])

        for i, eps in enumerate(epsilons):
            # HDC-maxent under central DP (noise the merged sum once)
            S_dp = privatize(S, eps, delta, rng, local_count=1)
            hdc_pdf = readout(S_dp, N, omega, grid)
            hdc_l1[i, s] = l1(hdc_pdf, truth, grid)
            v_h, _ = var_es(hdc_pdf, grid, 0.99)
            hdc_var[i, s] = abs(v_h - v_true)

            # DP histogram baseline
            his_pdf = dp_histogram(banks, grid, edges, eps, delta, rng)
            his_l1[i, s] = l1(his_pdf, truth, grid)
            v_b, _ = var_es(his_pdf, grid, 0.99)
            his_var[i, s] = abs(v_b - v_true)

    print(f"{'eps':>6}{'HDC L1':>10}{'Hist L1':>10}{'HDC VaRerr':>12}{'Hist VaRerr':>13}")
    for i, eps in enumerate(epsilons):
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
            ax.errorbar(epsilons, hdc.mean(1), yerr=hdc.std(1), fmt="r-s", capsize=3,
                        label="HDC maxent sketch")
            ax.errorbar(epsilons, his.mean(1), yerr=his.std(1), fmt="g-o", capsize=3,
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
        fig.savefig(OUTPUT_DIR / "fed_phase2_dp.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'fed_phase2_dp.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
