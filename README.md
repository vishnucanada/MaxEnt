# Federated Maximum-Entropy Distribution Estimation via Hypervector Sketches

A method for estimating a **global probability distribution across many data
holders without pooling their raw data** — in one communication round, with
differential privacy, at fixed per-holder cost — and reading a calibrated
density (including its tails / Value-at-Risk) back out via the maximum-entropy
principle.

This grew out of an exploration of Jaynes' Maximum-Entropy Principle (the
original MLE / fidelity work is preserved under [`legacy/`](legacy/)). The full
problem statement and validation plan is in
[`docs/proposal_federated_maxent_hdc.md`](docs/proposal_federated_maxent_hdc.md).

## Idea in one paragraph

Encode each scalar as a complex hypervector `psi(x)_j = exp(i ω_j x)` (a
Vector-Function-Architecture / Random-Fourier-Features encoding). Each data
holder keeps only a **bundle** `H = mean_x psi(x)` and a count — a fixed-size `M`
summary, regardless of how much data it has. Bundles **merge exactly**
(count-weighted mean), compose with the **Gaussian mechanism** for
(ε, δ)-differential privacy (because `|psi(x)_j| = 1`, sensitivity is bounded by
`√M`), and the server recovers a calibrated global density by a **maximum-entropy
readout** that treats the merged bundle as (relaxed) moment constraints.

## Validated results

| Claim | Result |
|---|---|
| Merge is lossless | Federated estimate == centralized, `L1 = 0.0000` |
| Communication | `2·M` floats / holder, independent of local size |
| No-DP accuracy (real ULB data) | 99% VaR within **0.5%** of empirical |
| Tail under privacy | Beats a DP-histogram on VaR error **2–4×** across ε |
| Method stability | Fourier-basis maxent stays well-conditioned where polynomial-moment maxent diverges past ~6 constraints |

A key finding baked into the code: under DP the maxent readout **must** be
relaxed (an L2 penalty on the multipliers, `reg>0`); exact moment matching is
unbounded on noisy moments.

## Repository structure

```
maxent/                     # the library (single source of truth)
  hdc.py                    #   VFA/RFF encoding, bundling, kernel readout
  maxent_fit.py             #   convex-dual maxent (exact + relaxed) and feature factories
  federated.py              #   sketch / merge / privatize / readout / VaR-ES / dp_histogram
  metrics.py                #   density distances (l1, ks)
  datasets.py               #   synthetic test distributions + non-IID consortium
  data.py                   #   filesystem paths
experiments/                # self-contained studies; each imports only from maxent/
  method_kde.py             #   bundling reproduces a KDE
  method_fourier_vs_poly.py #   Fourier vs polynomial maxent
  method_stability.py       #   accuracy-per-constraint / stability (key method result)
  federated_synthetic.py    #   H1 lossless merge, H2 communication, H4 non-IID
  federated_dp_sweep.py     #   H3 privacy sweep vs DP-histogram (synthetic)
  federated_creditcard.py   #   H1-H3 on real ULB credit-card Amount data
tests/                      # invariant tests (merge lossless, relaxation, etc.)
docs/                       # proposal / problem statement
legacy/                     # the original MLE / fidelity MaxEnt exploration (unused by the above)
data/                       # datasets (git-ignored)
outputs/                    # generated plots (git-ignored)
```

## Quickstart

```bash
pip install -r requirements.txt

# run a study (writes a plot to outputs/)
python -m experiments.federated_synthetic
python -m experiments.method_stability

# real-data experiment (needs data/creditcard.csv from Kaggle mlg-ulb/creditcardfraud)
python -m experiments.federated_creditcard

# tests (no pytest needed; pytest also works)
python -m tests.run
```

## Legacy

The original Maximum-Entropy exploration (deriving distributions by maximizing
log-likelihood or Bhattacharyya fidelity over an exponential family, plus the
least-squares basis reconstruction) lives in [`legacy/`](legacy/). It is kept for
reference and is independent of the federated contribution; see
[`legacy/README.md`](legacy/README.md).
