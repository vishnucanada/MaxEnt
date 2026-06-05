"""Federated Phase 3: real data (ULB credit-card transactions).

We estimate the cross-institution distribution of transaction Amount -- and its
99% VaR / Expected Shortfall -- from a non-IID partition of the real ULB dataset,
without pooling raw rows. Validates H1/H2/H4 and the H3 DP tail-accuracy claim on
real, heavy-tailed financial data.

Transform: z = standardize(log1p(Amount)) so the Fourier encoding is well-scaled;
results are mapped back to dollars for interpretation.

Requires data/creditcard.csv (Kaggle mlg-ulb/creditcardfraud).
Run:  python3.12 -m experiments.fed_phase3_creditcard
"""

import numpy as np
import pandas as pd

from maxent.hdc import sample_frequencies
from maxent.federated import institution_sketch, merge, privatize, readout, var_es
from maxent.data import DATA_DIR, OUTPUT_DIR
from experiments.fed_phase1_synth import l1
from experiments.fed_phase2_dp import dp_histogram


def load_amounts():
    a = pd.read_csv(DATA_DIR / "creditcard.csv", usecols=["Amount"])["Amount"].values
    la = np.log1p(a)
    mean, std = la.mean(), la.std()
    z = (la - mean) / std
    return z, (mean, std)


def z_to_dollars(z, transform):
    mean, std = transform
    return np.expm1(z * std + mean)


def noniid_partition(z, B, n_deciles, alpha, rng):
    """Assign each row to one of B banks, non-IID by amount-decile (Dirichlet)."""
    deciles = np.clip((np.searchsorted(np.quantile(z, np.linspace(0, 1, n_deciles + 1)), z) - 1),
                      0, n_deciles - 1)
    bank_of_row = np.empty(len(z), dtype=int)
    for d in range(n_deciles):
        idx = np.where(deciles == d)[0]
        w = rng.dirichlet(alpha * np.ones(B))      # this decile's split across banks
        bank_of_row[idx] = rng.choice(B, size=len(idx), p=w)
    return [z[bank_of_row == b] for b in range(B)]


def main():
    rng = np.random.default_rng(0)
    z, transform = load_amounts()
    B, M = 30, 32
    grid = np.linspace(-3, 6, 1200)

    # Empirical ground truth (we hold all data only to score against)
    truth = np.histogram(z, bins=grid, density=True)[0]
    truth = np.concatenate([truth, truth[-1:]])
    v_true_emp = np.quantile(z, 0.99)              # exact empirical VaR

    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)
    banks = noniid_partition(z, B, n_deciles=10, alpha=0.3, rng=rng)
    print(f"loaded {len(z)} transactions; {B} banks, sizes "
          f"{min(len(b) for b in banks)}..{max(len(b) for b in banks)}")

    # --- H1/H2: federated (no DP) vs centralized vs empirical ---
    # No DP -> moments are feasible -> exact maxent (reg=0) is near-exact.
    S, N = merge([institution_sketch(b, omega) for b in banks])
    fed_pdf = readout(S, N, omega, grid, reg=0.0)
    S_c, N_c = institution_sketch(z, omega)
    cen_pdf = readout(S_c, N_c, omega, grid, reg=0.0)

    v_fed, es_fed = var_es(fed_pdf, grid, 0.99)
    v_cen, _ = var_es(cen_pdf, grid, 0.99)
    print("\n=== H1: federated vs centralized vs empirical (no DP) ===")
    print(f"  L1(federated, centralized) = {l1(fed_pdf, cen_pdf, grid):.4f}")
    print(f"  L1(federated, empirical)   = {l1(fed_pdf, truth, grid):.4f}")
    print(f"  VaR99  empirical=${z_to_dollars(v_true_emp, transform):,.0f}  "
          f"federated=${z_to_dollars(v_fed, transform):,.0f}  "
          f"centralized=${z_to_dollars(v_cen, transform):,.0f}")
    print(f"  ES99   federated=${z_to_dollars(es_fed, transform):,.0f}")
    bytes_sketch = 2 * M * 8
    print(f"\n=== H2: {bytes_sketch} B/bank vs avg raw "
          f"{int(np.mean([len(b) for b in banks])) * 8} B "
          f"({np.mean([len(b) for b in banks]) / M / 2:.0f}x, independent of N) ===")

    # --- H3: DP sweep, federated maxent vs DP-histogram, VaR error in $ ---
    edges = np.linspace(-3, 6, 60)
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    delta = 1e-5
    print("\n=== H3: VaR99 error ($) vs privacy ===")
    print(f"{'eps':>6}{'HDC $err':>12}{'Hist $err':>12}")
    hdc_e, his_e = [], []
    for eps in epsilons:
        he, he2 = [], []
        for s in range(8):
            r = np.random.default_rng(s)
            # Under DP, use the minimal fixed reg that keeps the dual bounded
            # (not tuned per-epsilon).
            S_dp = privatize(S, eps, delta, r, local_count=1)
            v_h, _ = var_es(readout(S_dp, N, omega, grid, reg=1e-5), grid, 0.99)
            he.append(abs(z_to_dollars(v_h, transform) - z_to_dollars(v_true_emp, transform)))
            hp = dp_histogram(banks, grid, edges, eps, delta, r)
            v_b, _ = var_es(hp, grid, 0.99)
            he2.append(abs(z_to_dollars(v_b, transform) - z_to_dollars(v_true_emp, transform)))
        hdc_e.append(np.mean(he)); his_e.append(np.mean(he2))
        print(f"{eps:>6.1f}{np.mean(he):>12,.0f}{np.mean(he2):>12,.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs[0].plot(grid, truth, "k-", lw=2, label="empirical (all data)")
        axs[0].plot(grid, cen_pdf, "b--", label="centralized maxent")
        axs[0].plot(grid, fed_pdf, "r-", alpha=0.8, label="federated merge")
        axs[0].axvline(v_true_emp, color="k", ls=":", alpha=0.6)
        axs[0].axvline(v_fed, color="r", ls=":", alpha=0.6)
        axs[0].set_title("ULB Amount: federated vs centralized vs empirical")
        axs[0].set_xlabel("z = standardize(log1p(Amount))"); axs[0].set_ylabel("density"); axs[0].legend()
        axs[1].plot(epsilons, hdc_e, "r-s", label="HDC maxent sketch")
        axs[1].plot(epsilons, his_e, "g-o", label="DP histogram")
        axs[1].set_xscale("log"); axs[1].set_yscale("log")
        axs[1].set_xlabel("epsilon (smaller = more private)")
        axs[1].set_ylabel("VaR99 error ($)")
        axs[1].set_title("Tail-risk error vs privacy (real data)"); axs[1].legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fed_phase3_creditcard.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'fed_phase3_creditcard.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
