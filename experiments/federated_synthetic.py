"""Federated validation on a synthetic non-IID consortium (exact ground truth).

Tests H1 (federated merge is lossless vs centralized), H2 (constant
communication), H4 (non-IID mixture + tail recovered).

Run:  python3.12 -m experiments.federated_synthetic
"""

import numpy as np

from maxent import (
    sample_frequencies, institution_sketch, merge, readout, var_es, l1,
    make_consortium, consortium_true_pdf,
)
from maxent.data import OUTPUT_DIR

B, N_PER_BANK, M = 25, 2000, 32


def main():
    rng = np.random.default_rng(0)
    grid = np.linspace(-6, 10, 1000)

    banks, comps = make_consortium(B, N_PER_BANK, rng)
    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)

    # Federated: each bank sketches locally; server merges.
    S, N = merge([institution_sketch(d, omega) for d in banks])
    fed_pdf = readout(S, N, omega, grid)

    # Centralized oracle: one sketch over pooled data.
    S_c, N_c = institution_sketch(np.concatenate(banks), omega)
    cen_pdf = readout(S_c, N_c, omega, grid)

    truth = consortium_true_pdf(comps, grid)

    print("=== H1: federation vs centralized vs truth ===")
    print(f"  L1(federated, truth)   = {l1(fed_pdf, truth, grid):.4f}")
    print(f"  L1(centralized, truth) = {l1(cen_pdf, truth, grid):.4f}")
    print(f"  L1(federated, central) = {l1(fed_pdf, cen_pdf, grid):.4f}  (~0 => lossless)")

    v_t, e_t = var_es(truth, grid, 0.99)
    v_f, e_f = var_es(fed_pdf, grid, 0.99)
    print("\n=== Tail risk @ 99% ===")
    print(f"  true VaR={v_t:.3f} ES={e_t:.3f} | fed VaR={v_f:.3f} ES={e_f:.3f}"
          f"  (VaR err={abs(v_f - v_t):.3f}, ES err={abs(e_f - e_t):.3f})")

    bytes_sketch, bytes_raw = 2 * M * 8, N_PER_BANK * 8
    print("\n=== H2: communication per institution ===")
    print(f"  sketch={bytes_sketch} B, raw={bytes_raw} B -> "
          f"{bytes_raw / bytes_sketch:.0f}x, independent of N_k")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid, truth, "k-", lw=2, label="true global")
        ax.plot(grid, cen_pdf, "b--", label="centralized (oracle)")
        ax.plot(grid, fed_pdf, "r-", alpha=0.8, label="federated merge (no DP)")
        ax.axvline(v_t, color="k", ls=":", alpha=0.6, label="true VaR 99%")
        ax.set_title(f"Non-IID consortium of {B} banks: global distribution recovered federated")
        ax.set_xlabel("standardized log-amount")
        ax.set_ylabel("density")
        ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "federated_synthetic.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'federated_synthetic.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
