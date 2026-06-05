"""Federated validation on real data (ULB credit-card transactions).

Estimate the cross-institution distribution of transaction Amount -- and its 99%
VaR / Expected Shortfall -- from a *simulated* non-IID partition of the real ULB
dataset, without pooling raw rows. Validates H1/H2 and the H3 DP tail-accuracy
claim on real, moderately heavy-tailed financial data.

(For a *natural* federation by real bank id, see federated_ibm_aml.py.)

Requires data/creditcard.csv (Kaggle mlg-ulb/creditcardfraud).
Run:  python3.12 -m experiments.federated_creditcard
"""

import numpy as np
import pandas as pd

from maxent import federated_risk_study, print_study, plot_study
from maxent.data import DATA_DIR, OUTPUT_DIR

B, M = 30, 32


def load_amounts():
    a = pd.read_csv(DATA_DIR / "creditcard.csv", usecols=["Amount"])["Amount"].values
    la = np.log1p(a)
    mean, std = la.mean(), la.std()
    return (la - mean) / std, (mean, std)


def noniid_partition(z, n_banks, n_deciles, alpha, rng):
    """Simulate a non-IID federation: assign rows to banks by amount-decile (Dirichlet)."""
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
    z, (mean, std) = load_amounts()
    banks = noniid_partition(z, B, n_deciles=10, alpha=0.3, rng=rng)
    to_usd = lambda zz: np.expm1(np.asarray(zz) * std + mean)
    print(f"ULB creditcard: {B} simulated banks, "
          f"sizes {min(map(len, banks))}..{max(map(len, banks))}")

    res = federated_risk_study(banks, M=M, bandwidth=1.0,
                               grid=np.linspace(-3, 6, 1200))
    print_study(res, to_usd=to_usd, unit="$")

    plot_study(res, OUTPUT_DIR / "federated_creditcard.png", to_usd=to_usd,
               title="ULB Amount: federated vs centralized vs empirical",
               xlabel="z = standardize(log1p(Amount))")
    print(f"\nsaved plot -> {OUTPUT_DIR / 'federated_creditcard.png'}")


if __name__ == "__main__":
    main()
