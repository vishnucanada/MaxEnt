"""Federated validation on real data (ULB credit-card transactions).

Estimate the cross-institution distribution of transaction Amount -- and its 99%
VaR / Expected Shortfall -- from a non-IID partition of the real ULB dataset,
without pooling raw rows. Validates H1/H2 and the H3 DP tail-accuracy claim on
real, heavy-tailed financial data.

reg policy: 0 with no DP (feasible moments -> exact, near-perfect VaR);
            a small fixed reg under DP (minimal stabilizing, not tuned per-eps).

Requires data/creditcard.csv (Kaggle mlg-ulb/creditcardfraud).
Run:  python3.12 -m experiments.federated_creditcard
"""

import numpy as np
import pandas as pd

from maxent import (
    sample_frequencies, institution_sketch, merge, privatize, readout, var_es,
    dp_histogram, l1,
)
from maxent.data import DATA_DIR, OUTPUT_DIR

B, M = 30, 32
DP_REG = 1e-5


def load_amounts():
    a = pd.read_csv(DATA_DIR / "creditcard.csv", usecols=["Amount"])["Amount"].values
    la = np.log1p(a)
    mean, std = la.mean(), la.std()
    return (la - mean) / std, (mean, std)


def z_to_dollars(z, transform):
    mean, std = transform
    return np.expm1(z * std + mean)


def noniid_partition(z, n_banks, n_deciles, alpha, rng):
    """Assign each row to one of n_banks banks, non-IID by amount-decile (Dirichlet)."""
    deciles = np.clip(np.searchsorted(np.quantile(z, np.linspace(0, 1, n_deciles + 1)), z) - 1,
                      0, n_deciles - 1)
    bank_of_row = np.empty(len(z), dtype=int)
    for d in range(n_deciles):
        idx = np.where(deciles == d)[0]
        w = rng.dirichlet(alpha * np.ones(n_banks))
        bank_of_row[idx] = rng.choice(n_banks, size=len(idx), p=w)
    return [z[bank_of_row == b] for b in range(n_banks)]


def main():
    rng = np.random.default_rng(0)
    z, transform = load_amounts()
    grid = np.linspace(-3, 6, 1200)

    truth = np.histogram(z, bins=grid, density=True)[0]
    truth = np.concatenate([truth, truth[-1:]])
    v_true = np.quantile(z, 0.99)                  # exact empirical VaR
    to_usd = lambda zz: z_to_dollars(zz, transform)

    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)
    banks = noniid_partition(z, B, n_deciles=10, alpha=0.3, rng=rng)
    print(f"loaded {len(z)} transactions; {B} banks, sizes "
          f"{min(len(b) for b in banks)}..{max(len(b) for b in banks)}")

    # H1/H2: no DP -> exact maxent (reg=0)
    S, N = merge([institution_sketch(b, omega) for b in banks])
    fed_pdf = readout(S, N, omega, grid, reg=0.0)
    S_c, N_c = institution_sketch(z, omega)
    cen_pdf = readout(S_c, N_c, omega, grid, reg=0.0)
    v_fed, es_fed = var_es(fed_pdf, grid, 0.99)
    v_cen, _ = var_es(cen_pdf, grid, 0.99)

    print("\n=== H1: federated vs centralized vs empirical (no DP) ===")
    print(f"  L1(federated, centralized) = {l1(fed_pdf, cen_pdf, grid):.4f}")
    print(f"  VaR99 empirical=${to_usd(v_true):,.0f}  federated=${to_usd(v_fed):,.0f}  "
          f"centralized=${to_usd(v_cen):,.0f}   ES99 fed=${to_usd(es_fed):,.0f}")
    avg_n = int(np.mean([len(b) for b in banks]))
    print(f"\n=== H2: {2 * M * 8} B/bank vs avg raw {avg_n * 8} B "
          f"({avg_n / M / 2:.0f}x, independent of N) ===")

    # H3: DP sweep, VaR error in dollars
    edges = np.linspace(-3, 6, 60)
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    print("\n=== H3: VaR99 error ($) vs privacy ===")
    print(f"{'eps':>6}{'HDC $err':>12}{'Hist $err':>12}")
    hdc_e, his_e = [], []
    for eps in epsilons:
        he, he2 = [], []
        for s in range(8):
            r = np.random.default_rng(s)
            v_h, _ = var_es(readout(privatize(S, eps, 1e-5, r), N, omega, grid, reg=DP_REG), grid, 0.99)
            he.append(abs(to_usd(v_h) - to_usd(v_true)))
            v_b, _ = var_es(dp_histogram(banks, grid, edges, eps, 1e-5, r), grid, 0.99)
            he2.append(abs(to_usd(v_b) - to_usd(v_true)))
        hdc_e.append(np.mean(he))
        his_e.append(np.mean(he2))
        print(f"{eps:>6.1f}{np.mean(he):>12,.0f}{np.mean(he2):>12,.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs[0].plot(grid, truth, "k-", lw=2, label="empirical (all data)")
        axs[0].plot(grid, cen_pdf, "b--", label="centralized maxent")
        axs[0].plot(grid, fed_pdf, "r-", alpha=0.8, label="federated merge")
        axs[0].axvline(v_true, color="k", ls=":", alpha=0.6)
        axs[0].set_title("ULB Amount: federated vs centralized vs empirical")
        axs[0].set_xlabel("z = standardize(log1p(Amount))")
        axs[0].set_ylabel("density")
        axs[0].legend()
        axs[1].plot(epsilons, hdc_e, "r-s", label="HDC maxent sketch")
        axs[1].plot(epsilons, his_e, "g-o", label="DP histogram")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("epsilon (smaller = more private)")
        axs[1].set_ylabel("VaR99 error ($)")
        axs[1].set_title("Tail-risk error vs privacy (real data)")
        axs[1].legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "federated_creditcard.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'federated_creditcard.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
