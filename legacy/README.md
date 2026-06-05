# Legacy: original Maximum-Entropy exploration

This directory preserves the project's **original** method, kept for reference.
It is **not used** by the current federated maxent-HDC contribution (`maxent/`).

It derives a probability distribution of the exponential-family form
`p(x) = (1/Z) exp(Σ λ_i f_i(x))` by either maximizing log-likelihood (MLE) or
maximizing Bhattacharyya fidelity, with `f_i` the first few polynomial moments.

Modules:
- `core.py` — exp-family pdf, moment/normalization constraints, `fit(objective=...)`
- `metrics.py` — fidelity, entropy, rmse, histogram probabilities
- `plotting.py` — matplotlib helpers
- `io.py` — CSV / dataframe helpers
- experiments: `combined.py` (primary MLE+fidelity fit), `basis.py` (least-squares
  basis reconstruction), `visualizer.py` (Dash slider app), `dfft.py`,
  `maclaurin.py`, `sampling_data.py`, `deriving_uniform_distribution/lagrange.py`

Run from the repo root, e.g.:

```bash
python -m legacy.combined
```

Legacy experiments need extra dependencies (scikit-learn, seaborn, dash, plotly,
sympy); see `legacy/requirements.txt`.
