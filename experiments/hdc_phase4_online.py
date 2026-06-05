"""Phase 4 test: online tracking of a drifting distribution.

A stream whose distribution changes partway through (unimodal -> shifts ->
becomes bimodal). We compare three estimators at snapshots in time:
  * cumulative bundle (no forgetting)   -- should smear across the drift
  * forgetting bundle (EWMA) maxent     -- should track the current regime
  * the true current pdf

Run:  python3.12 -m experiments.hdc_phase4_online
"""

import numpy as np

from maxent.hdc import sample_frequencies
from maxent.streaming import OnlineBundle
from maxent.data import OUTPUT_DIR
from experiments.hdc_phase1_kde import l1_error


def regime_pdf(t_frac, x):
    """The true pdf as a function of stream progress t_frac in [0, 1]."""
    if t_frac < 0.5:
        mu = -2 + 4 * t_frac            # mean drifts -2 -> 0
        return np.exp(-0.5 * ((x - mu) / 0.7) ** 2) / (0.7 * np.sqrt(2 * np.pi))
    # second half: bimodal
    a = np.exp(-0.5 * ((x + 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
    b = np.exp(-0.5 * ((x - 2) / 0.6) ** 2) / (0.6 * np.sqrt(2 * np.pi))
    return 0.5 * a + 0.5 * b


def regime_sample(t_frac, rng):
    if t_frac < 0.5:
        mu = -2 + 4 * t_frac
        return rng.normal(mu, 0.7)
    return rng.normal(-2, 0.6) if rng.random() < 0.5 else rng.normal(2, 0.6)


def main():
    rng = np.random.default_rng(0)
    STREAM = 6000
    N_FREQ = 8
    grid = np.linspace(-6, 6, 600)

    omega = sample_frequencies(N_FREQ, bandwidth=1.5, kind="gaussian", seed=3)
    cumulative = OnlineBundle(omega, forget=None)
    forgetting = OnlineBundle(omega, forget=0.02)   # memory window ~50 samples

    snapshots = [0.25, 0.49, 0.75, 1.0]
    snap_idx = {int(s * STREAM) - 1: s for s in snapshots}
    captured = {}

    cum_err, fgt_err = [], []
    for t in range(STREAM):
        t_frac = (t + 1) / STREAM
        x = regime_sample(t_frac, rng)
        cumulative.update(x)
        forgetting.update(x)

        if t in snap_idx:
            truth = regime_pdf(t_frac, grid)
            cum_pdf = cumulative.density_maxent(grid)
            fgt_pdf = forgetting.density_maxent(grid)
            captured[snap_idx[t]] = (truth, cum_pdf, fgt_pdf)
            cum_err.append(l1_error(cum_pdf, truth, grid))
            fgt_err.append(l1_error(fgt_pdf, truth, grid))

    print(f"{'progress':>10}{'cumulative L1':>16}{'forgetting L1':>16}")
    for s, ec, ef in zip(snapshots, cum_err, fgt_err):
        print(f"{s:>10.2f}{ec:>16.4f}{ef:>16.4f}")
    print(f"\nmean: cumulative={np.mean(cum_err):.4f}  forgetting={np.mean(fgt_err):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        for ax, s in zip(axs, snapshots):
            truth, cum_pdf, fgt_pdf = captured[s]
            ax.plot(grid, truth, "k-", lw=2, label="true (current)")
            ax.plot(grid, cum_pdf, "b--", label="cumulative bundle")
            ax.plot(grid, fgt_pdf, "r-", alpha=0.85, label="forgetting bundle")
            ax.set_title(f"stream progress {s:.0%}")
            ax.legend(fontsize=8)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "hdc_phase4_online.png", dpi=110)
        print(f"\nsaved plot -> {OUTPUT_DIR / 'hdc_phase4_online.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
