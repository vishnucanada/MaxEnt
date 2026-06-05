"""Federated validation on the IBM AML dataset -- a NATURAL bank federation.

Unlike the credit-card experiment (which simulates a partition), here each
institution is a real bank: we partition transactions by the dataset's own
`From Bank` column. We estimate the cross-bank distribution of transaction
amount -- and its 99% VaR -- federated and under DP, without pooling raw rows.

Requires data/Money-Laundering-IBM/HI-Small_Trans.csv (Kaggle
ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
Run:  python3.12 -m experiments.federated_ibm_aml
"""

import numpy as np
import pandas as pd

from maxent import federated_risk_study, print_study, plot_study
from maxent.data import DATA_DIR, OUTPUT_DIR

IBM_FILE = "Money-Laundering-IBM/HI-Small_Trans.csv"
TOP_K_BANKS = 100
M = 64
# Heavy-tailed amounts span many orders of magnitude; a finer kernel
# (larger frequency bandwidth) is needed to resolve the upper-tail quantile.
BANDWIDTH = 2.0


def load_consortium():
    """Top-K banks by transaction volume, partitioned by real `From Bank`.

    Returns (banks, to_usd) where banks is a list of 1-D arrays in the
    standardized log-amount domain and to_usd maps that domain back to dollars.
    """
    df = pd.read_csv(DATA_DIR / IBM_FILE, usecols=["From Bank", "Amount Paid"])
    top = df["From Bank"].value_counts().head(TOP_K_BANKS).index
    df = df[df["From Bank"].isin(top)].copy()

    la = np.log1p(df["Amount Paid"].values)
    mean, std = la.mean(), la.std()
    df["z"] = (la - mean) / std
    banks = [g["z"].values for _, g in df.groupby("From Bank")]

    def to_usd(z):
        return np.expm1(np.asarray(z) * std + mean)

    return banks, to_usd


def main():
    banks, to_usd = load_consortium()
    print(f"IBM HI-Small: {TOP_K_BANKS} real banks, "
          f"sizes {min(map(len, banks))}..{max(map(len, banks))}")

    res = federated_risk_study(banks, M=M, bandwidth=BANDWIDTH)
    print_study(res, to_usd=to_usd, unit="$")

    plot_study(res, OUTPUT_DIR / "federated_ibm_aml.png", to_usd=to_usd,
               title=f"IBM AML (HI-Small): {TOP_K_BANKS}-bank natural federation",
               xlabel="z = standardize(log1p(Amount Paid))")
    print(f"\nsaved plot -> {OUTPUT_DIR / 'federated_ibm_aml.png'}")


if __name__ == "__main__":
    main()
