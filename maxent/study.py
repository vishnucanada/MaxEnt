"""Reusable federated risk-distribution study (shared by the real-data experiments).

Given a list of per-institution 1-D arrays (already in a standardized domain),
run the federated pipeline end to end: lossless-merge check, communication, and
a differential-privacy sweep of VaR error vs a DP-histogram baseline. Keeps the
experiment scripts thin -- they only load + partition + transform data.
"""

import numpy as np

from .hdc import sample_frequencies
from .federated import (
    institution_sketch, merge, privatize, readout, var_es, dp_histogram,
)
from .metrics import l1

DEFAULT_EPSILONS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def federated_risk_study(banks, M=32, bandwidth=1.0, grid=None, n_edges=60,
                         level=0.99, epsilons=DEFAULT_EPSILONS, delta=1e-5,
                         dp_reg=1e-5, seeds=8, seed_omega=11):
    """Run the study. `banks` is a list of 1-D arrays in a standardized domain.

    Returns a results dict; VaR values are in the standardized domain (map to
    natural units with your inverse transform).
    """
    pooled = np.concatenate(banks)
    if grid is None:
        lo = pooled.min() - 0.5
        hi = np.quantile(pooled, 0.9999) + 0.5
        grid = np.linspace(lo, hi, 1200)
    edges = np.linspace(grid[0], grid[-1], n_edges)
    omega = sample_frequencies(M, bandwidth, "gaussian", seed_omega)

    # H1/H2: no DP -> exact maxent (reg=0)
    sketches = [institution_sketch(b, omega) for b in banks]
    S, N = merge(sketches)
    fed_pdf = readout(S, N, omega, grid, reg=0.0)
    S_c, N_c = institution_sketch(pooled, omega)
    cen_pdf = readout(S_c, N_c, omega, grid, reg=0.0)

    truth = np.histogram(pooled, bins=grid, density=True)[0]
    truth = np.append(truth, truth[-1])
    v_emp = float(np.quantile(pooled, level))

    # H3: DP sweep (per-seed VaR values, in the standardized domain)
    sweep = []
    for eps in epsilons:
        hdc_v, his_v = [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            hdc_v.append(var_es(readout(privatize(S, eps, delta, rng), N, omega, grid, reg=dp_reg), grid, level)[0])
            his_v.append(var_es(dp_histogram(banks, grid, edges, eps, delta, rng), grid, level)[0])
        sweep.append((eps, np.array(hdc_v), np.array(his_v)))

    return {
        "grid": grid, "truth": truth, "fed_pdf": fed_pdf, "cen_pdf": cen_pdf,
        "v_emp": v_emp,
        "v_fed": var_es(fed_pdf, grid, level)[0],
        "v_cen": var_es(cen_pdf, grid, level)[0],
        "es_fed": var_es(fed_pdf, grid, level)[1],
        "N": N, "M": M, "n_banks": len(banks),
        "l1_fed_cen": l1(fed_pdf, cen_pdf, grid),
        "sweep": sweep,
    }


def print_study(res, to_usd=lambda z: z, unit="$"):
    """Print the H1/H2/H3 summary, mapping VaR to natural units via `to_usd`."""
    print(f"  banks={res['n_banks']}  pooled N={res['N']}")
    print("\n=== H1: federated vs centralized vs empirical (no DP) ===")
    print(f"  L1(federated, centralized) = {res['l1_fed_cen']:.4f}")
    print(f"  VaR  empirical={unit}{to_usd(res['v_emp']):,.0f}  "
          f"federated={unit}{to_usd(res['v_fed']):,.0f}  "
          f"centralized={unit}{to_usd(res['v_cen']):,.0f}   "
          f"ES fed={unit}{to_usd(res['es_fed']):,.0f}")
    bytes_sketch = 2 * res["M"] * 8
    avg_n = res["N"] / res["n_banks"]
    print(f"\n=== H2: {bytes_sketch} B/bank, ~{avg_n / res['M'] / 2:.0f}x vs avg raw, independent of N ===")
    print("\n=== H3: VaR error vs privacy ===")
    print(f"{'eps':>6}{'HDC err':>14}{'Hist err':>14}")
    for eps, hdc_v, his_v in res["sweep"]:
        e_h = np.mean(np.abs(to_usd(hdc_v) - to_usd(res["v_emp"])))
        e_b = np.mean(np.abs(to_usd(his_v) - to_usd(res["v_emp"])))
        print(f"{eps:>6.1f}{unit + format(e_h, ',.0f'):>14}{unit + format(e_b, ',.0f'):>14}")


def plot_study(res, outpath, to_usd=lambda z: z, title="", xlabel="standardized value"):
    """Two-panel figure: density (federated/centralized/empirical) + VaR-error vs eps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = res["grid"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(grid, res["truth"], "k-", lw=2, label="empirical (all data)")
    axs[0].plot(grid, res["cen_pdf"], "b--", label="centralized maxent")
    axs[0].plot(grid, res["fed_pdf"], "r-", alpha=0.8, label="federated merge")
    axs[0].axvline(res["v_emp"], color="k", ls=":", alpha=0.6)
    axs[0].set_title(title)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("density")
    axs[0].legend()

    eps = [s[0] for s in res["sweep"]]
    hdc_e = [np.mean(np.abs(to_usd(s[1]) - to_usd(res["v_emp"]))) for s in res["sweep"]]
    his_e = [np.mean(np.abs(to_usd(s[2]) - to_usd(res["v_emp"]))) for s in res["sweep"]]
    axs[1].plot(eps, hdc_e, "r-s", label="HDC maxent sketch")
    axs[1].plot(eps, his_e, "g-o", label="DP histogram")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("epsilon (smaller = more private)")
    axs[1].set_ylabel("VaR error")
    axs[1].set_title("Tail-risk error vs privacy")
    axs[1].legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
