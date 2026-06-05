# Federated, Privacy-Preserving Distribution Estimation via Mergeable Maximum-Entropy Hypervector Sketches
### Application: cross-institution financial risk & AML analytics

*Problem statement & validation plan — draft v2 (finance-anchored)*

## 1. The real-world problem

Financial supervisors and bank consortia repeatedly need the **distribution** of a
quantity *across institutions* — not a single average, the whole shape: the
industry-wide distribution of transaction amounts, credit-risk scores,
counterparty exposures, or fraud-indicator scores. The shape is the decision:
**Value-at-Risk and Expected Shortfall are tail quantiles of a loss
distribution**; systemic risk is the aggregate distribution of exposures;
money-laundering and fraud signatures live in the tails and in cross-bank
patterns no single institution sees.

Three constraints collide:
1. **Legal data-sharing barriers.** Raw customer transaction data cannot be
   pooled — banking secrecy, GDPR/GLBA, and competition law. This is not
   hypothetical: **Transaction Monitoring Netherlands (TMNL)**, a 2020
   data-pooling utility of five Dutch banks, was **wound down (~2023) over
   privacy-law concerns**.
2. **Active demand for a privacy-preserving alternative.** **MAS COSMIC**
   (Singapore) and the **BIS Project Aurora** explicitly pursue
   privacy-enhancing cross-institution AML analytics; Aurora found **local
   differential privacy** among the best-performing approaches.
3. **Scale & operational cost.** Cross-border payment volumes are enormous;
   periodic manual regulatory reporting of a few summary statistics is coarse
   and slow.

The supervisor/consortium wants the **global distribution** (including its
**tails**) computed **without centralizing raw data**, in **one communication
round**, with a **formal (ε, δ)-differential-privacy guarantee**, at **bounded
per-institution memory**.

## 2. Why existing approaches fall short (the precise gap)

- **Data pooling (TMNL-style)** delivers the full joint picture but is exactly
  what privacy law blocks.
- **Federated *learning* for AML/fraud** (FL classifiers, the dominant academic
  line) shares model updates to train a *detector* — it does **not** recover a
  calibrated cross-institution *distribution*, and iterative training needs many
  rounds.
- **Homomorphic-encryption / MPC pipelines** (e.g. Aurora variants) are powerful
  but heavy and many-round; overkill when the deliverable is a marginal
  distribution.
- **Regulatory summary reporting** gives means and a few fixed quantiles, not the
  density — and crucially not principled **tail** estimates, which is what risk
  actually needs.

**The opening:** use the HDC bundle as a *mergeable, DP-composable density
sketch*, and recover the calibrated global density — **including a principled
tail** — with a **maximum-entropy readout**. The maxent step turns a small,
low-sensitivity moment vector into a full distribution, so we get strong privacy
(few bounded-sensitivity numbers) *and* a usable risk object (VaR/ES quantiles,
tail probabilities, entropy) in **one round** — which neither FL-detectors,
HE/MPC pipelines, nor summary reporting provide.

## 3. Thesis / central claim

> A consortium of institutions can estimate its **global distribution of a risk
> quantity** — and its **tail** (VaR / Expected Shortfall) — in a **single
> round**, each institution transmitting one fixed-size **maximum-entropy
> hypervector sketch**. These sketches **merge exactly**, compose with the
> **Gaussian mechanism for (ε, δ)-DP**, cost **O(M) per institution independent
> of local sample count**, and yield a **calibrated density** — matching
> centralized estimation and beating DP-histogram baselines, especially in the
> **tails** and under **non-IID** institution heterogeneity.

## 4. Method sketch

**Encoding (per institution, streaming, O(M) memory).** Fix shared random
frequencies ω ∈ ℝ^M (public seed). Each value x (e.g. a transaction amount or
risk score) is encoded as ψ(x)ⱼ = exp(i·ωⱼ·x). Institution k keeps only a running
bundle and a count:

    H_k = (1/N_k) Σ_{x at institution k} ψ(x),     N_k.

H_k is the empirical characteristic function of institution k's data at
frequencies ω — a fixed-size-M sufficient statistic regardless of N_k.

**Exact merge (one round).** Each H_k is a count-weighted mean, so

    H_global = ( Σ_k N_k H_k ) / ( Σ_k N_k )

is *exactly* the bundle centralized data would have produced. Merging is
associative/commutative → hierarchical aggregation, stragglers, partial merges
all well-defined; pairs naturally with **secure aggregation** so the server never
sees an individual H_k.

**Differential privacy (bounded sensitivity).** Every component |ψ(x)ⱼ| = 1, so
one record changes the unnormalized institution sum by L2-norm ≤ √M. The
**Gaussian mechanism** then gives (ε, δ)-DP on H_k (local DP) or on the merged
sum (central DP, with secure aggregation). Privacy budget scales with M (a few
hundred), **not** with histogram resolution — the key efficiency, and the reason
tails survive privatization.

**Maximum-entropy readout (server side).** Treat [Re(H_global), Im(H_global)] as
(noisy, relaxed) moment targets and fit

    p(x) = (1/Z) exp( Σⱼ aⱼ cos(ωⱼ x) + bⱼ sin(ωⱼ x) )

via the convex dual (`maxent/maxent_fit.py`). DP noise → naturally relaxed
constraints, which the dual already tolerates. Output: a calibrated global
density → industry VaR/ES, tail probabilities, quantiles, entropy.

*(Already built & validated: VFA encoding/bundling `maxent/hdc.py`; convex-dual
maxent + Fourier features `maxent/maxent_fit.py`; online forgetting bundle
`maxent/streaming.py`. The federated layer = merge + DP noise on top.)*

## 5. Hypotheses to validate (falsifiable)

- **H1 — Federation is lossless.** One-round merged estimate ≈ centralized
  HDC-maxent on pooled data (no accuracy gap from federating).
- **H2 — Communication is constant.** Target accuracy reached with M numbers per
  institution per round, independent of N_k; orders of magnitude below raw data
  or fine histograms.
- **H3 — Tails survive privacy, and beat DP-histograms.** Under (ε, δ)-DP,
  HDC-maxent dominates DP-histogram on **tail / VaR / Expected-Shortfall** error
  at matched ε, because it perturbs few low-sensitivity moments, not many bins.
- **H4 — Non-IID is handled natively.** When institutions hold different local
  distributions (different customer bases), the merged bundle forms the true
  global **mixture**; maxent recovers multimodal industry densities that
  polynomial-moment methods cannot (our validated stability result).

## 6. Minimal experiment plan

**Datasets** (partition rows across simulated institutions to form the fleet)
- *Synthetic non-IID consortium* (controlled): each bank a different mixture of
  amount/score components; sweep heterogeneity, #banks, samples/bank. Ground
  truth known → exact L1/KS and exact tail/VaR error.
- *Credit-card fraud* (ULB/Kaggle): heavy-tailed transaction amounts → a direct
  tail-estimation testbed.
- *PaySim / AMLSim* (synthetic mobile-money & AML simulators): naturally support
  multi-institution partitions and injected laundering patterns → non-IID + rare
  tails.
- *Lending Club*: credit-risk scores across partitions for a credit-risk
  distribution.

**Baselines**
- Centralized HDC-maxent (oracle upper bound).
- DP federated **histogram** (the standard federated-analytics baseline).
- Merged quantile sketches (t-digest / KLL) — quantile/VaR error only.
- DP federated **polynomial-moment** maxent (isolates the Fourier+stability win).

**Metrics**
- Density L1 / KS to true global; **VaR & Expected-Shortfall error** at the
  95/99% level; quantile error.
- Communication bytes/institution; per-institution memory.
- Privacy–utility curves (accuracy & VaR error vs ε); scaling vs #institutions;
  vs non-IID severity.

**Ablations**: M (sketch size) vs accuracy & privacy; frequency distribution /
kernel bandwidth; local-DP vs central-DP-with-secure-aggregation.

## 7. Honest risks & limitations

- **Dimensionality.** Cleanest for 1-D / low-D marginals (a risk score, an
  amount). Joint multivariate risk needs random *vector* frequencies and larger
  M; scope the headline to marginals + low-D, flag high-D as future work.
- **Local-DP noise.** Per-institution noise is much harsher than
  central-DP-with-secure-aggregation; state the trust model explicitly and report
  both.
- **Tail vs kernel artifacts.** Smooth Fourier kernels ring at hard supports
  (seen in Phase 1 on the exponential); since the *tail* is the product here,
  evaluate it carefully and consider support warping / one-sided kernels.
- **Novelty boundary.** Bundle = kernel mean embedding (RFF) is known; the
  contribution is the **federated-merge + DP + maxent-tail-density** combination
  and its privacy/communication efficiency. Keep that framing precise.

## 8. Why this is a defensible "big claim"

A single, crisp, testable contribution that a regulator or bank consortium would
actually want: **one-round, communication-O(M), differentially-private estimation
of the cross-institution risk distribution — including a principled tail
(VaR/ES)** — with exact mergeability and a maximum-entropy readout. It fills a
real, demonstrated gap (the data-pooling route was legally shut down; current
alternatives do detection/learning, not private distribution estimation). It
plants the project's maximum-entropy core (Jaynes: entropy bridges physics and
information theory) in a concrete financial-information system, and turns the
validated stability result (Fourier-maxent stays well-conditioned where
polynomial-moment maxent diverges) into a direct **tail-accuracy-under-privacy**
advantage.
```
References for motivation (verify before citing in a paper):
TMNL wind-down over privacy law; MAS COSMIC; BIS Project Aurora (PETs + local DP for AML);
federated-learning-for-AML literature; ULB credit-card fraud; PaySim / AMLSim simulators.
```
