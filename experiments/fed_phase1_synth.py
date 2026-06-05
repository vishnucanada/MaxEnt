"""Federated Phase 1: non-IID bank consortium, exact ground truth.

B banks each hold a different mixture of a 'normal' transaction component and a
small-weight 'large/rare' tail component (the risk tail). We compare the global
distribution recovered by:
  * centralized HDC-maxent on pooled data        (oracle)
  * federated merge of per-bank sketches (no DP)  (our method)
  * the analytic true global mixture              (ground truth)

Tests H1 (federation lossless), H2 (constant communication), H4 (non-IID
mixture recovered). Data is in a standardized 'log-amount' domain; the real
creditcardfraud Amount column drops into the same pipeline via a log+z transform.

Run:  python3.12 -m experiments.fed_phase1_synth
"""

import numpy as np

from maxent.hdc import sample_frequencies
from maxent.federated import institution_sketch, merge, readout, var_es
from maxent.data import OUTPUT_DIR


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def make_consortium(B, n_per_bank, rng):
    """Each bank: main component N(m_b, 0.8) + rare tail N(5, 1.2) weight q_b."""
    banks, comps = [], []
    for _ in range(B):
        m_b = rng.uniform(-1.5, 1.5)          # bank-specific 'normal' center (non-IID)
        q_b = rng.uniform(0.01, 0.05)         # rare large-transaction weight
        n = n_per_bank
        n_tail = rng.binomial(n, q_b)
        data = np.concatenate([
            rng.normal(m_b, 0.8, n - n_tail),
            rng.normal(5.0, 1.2, n_tail),
        ])
        banks.append(data)
        comps.append((n, m_b, 0.8, q_b, 5.0, 1.2))
    return banks, comps


def true_global_pdf(comps, grid):
    N = sum(c[0] for c in comps)
    p = np.zeros_like(grid)
    for n, m_b, s_b, q_b, mt, st in comps:
        p += (n / N) * ((1 - q_b) * gauss(grid, m_b, s_b) + q_b * gauss(grid, mt, st))
    return p


def l1(p, q, grid):
    return np.trapz(np.abs(p - q), grid)


def main():
    rng = np.random.default_rng(0)
    B = 25
    N_PER_BANK = 2000
    M = 32                      # frequencies -> 64 constraints; 2M floats per sketch
    grid = np.linspace(-6, 10, 1000)

    banks, comps = make_consortium(B, N_PER_BANK, rng)
    omega = sample_frequencies(M, bandwidth=1.0, kind="gaussian", seed=11)

    # --- federated: each bank sketches locally, server merges ---
    sketches = [institution_sketch(d, omega) for d in banks]
    S, N = merge(sketches)
    fed_pdf = readout(S, N, omega, grid)

    # --- centralized oracle: pool all data, one sketch ---
    pooled = np.concatenate(banks)
    S_c, N_c = institution_sketch(pooled, omega)
    cen_pdf = readout(S_c, N_c, omega, grid)

    # --- ground truth ---
    truth = true_global_pdf(comps, grid)

    # H1: federation lossless
    print("=== H1: federation vs centralized vs truth ===")
    print(f"  L1(federated, truth)   = {l1(fed_pdf, truth, grid):.4f}")
    print(f"  L1(centralized, truth) = {l1(cen_pdf, truth, grid):.4f}")
    print(f"  L1(federated, central) = {l1(fed_pdf, cen_pdf, grid):.4f}  (~0 => lossless)")

    # Tail risk (the deliverable)
    v_t, e_t = var_es(truth, grid, 0.99)
    v_f, e_f = var_es(fed_pdf, grid, 0.99)
    print("\n=== Tail risk @ 99% ===")
    print(f"  true   VaR={v_t:.3f}  ES={e_t:.3f}")
    print(f"  fed    VaR={v_f:.3f}  ES={e_f:.3f}   (err VaR={abs(v_f-v_t):.3f}, ES={abs(e_f-e_t):.3f})")

    # H2: communication
    bytes_sketch = 2 * M * 8           # M complex -> 2M float64
    bytes_raw = N_PER_BANK * 8
    print("\n=== H2: communication per institution ===")
    print(f"  sketch = {bytes_sketch} B (2*M floats), raw = {bytes_raw} B")
    print(f"  compression = {bytes_raw / bytes_sketch:.0f}x, independent of N_k")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid, truth, "k-", lw=2, label="true global")
        ax.plot(grid, cen_pdf, "b--", label="centralized (oracle)")
        ax.plot(grid, fed_pdf, "r-", alpha=0.8, label="federated merge (no DP)")
        ax.axvline(v_t, color="k", ls=":", alpha=0.6, label="true VaR 99%")
        ax.axvline(v_f, color="r", ls=":", alpha=0.6, label="fed VaR 99%")
        ax.set_title(f"Non-IID consortium of {B} banks: global distribution recovered federated")
        ax.set_xlabel("standardized log-amount")
        ax.set_ylabel("density")
        ax.legend()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fed_phase1_synth.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'fed_phase1_synth.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
