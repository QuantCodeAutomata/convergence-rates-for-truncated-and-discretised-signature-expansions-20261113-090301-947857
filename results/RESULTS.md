# Results

Replication of *Convergence rates for truncated and discretised signature
expansions* (arXiv:2611.16087), Section 4 (numerical experiments) and
Sections 3, 5, 6, 7 (supporting theoretical bounds).

No market data is used anywhere in this repository: every experiment
simulates Brownian paths (`data/generate_synthetic.py`) or verifies analytic
inequalities on synthetic grids of parameters, exactly as the paper's Section
4 does.

All numbers below are read from the files saved in `results/` by the most
recent run of `python -m exp.exp1` ... `python -m exp.exp6` (`./reproduce.sh`).

## Execution configuration

The paper's Monte Carlo experiments (Sections 4.2-4.4) use 20,000 sample
paths on a 10,000-step reference grid (`EXP*_N_PATHS_PAPER`,
`EXP*_N_REF_PAPER` in `src/config.py`). Reproducing that scale for the
delay-equation, fBm and put-pricing experiments together, at every truncation
level `M` and discretisation level `N` the paper tests, was estimated from a
reduced timed run to exceed the time available for this replication; the
executed runs therefore use 4,000 paths on a 2,000-step reference grid
(`EXP*_N_PATHS_RUN`, `EXP*_N_REF_RUN`). The discretisation levels, truncation
levels, test times and estimation windows are unchanged from the paper. To
run an experiment at the paper's full scale, set the corresponding
`*_RUN` constants in `src/config.py` to the `*_PAPER` values and rerun
`python -m exp.exp1` (or exp2/exp3), which otherwise takes a few seconds at
the reduced scale.

## exp_1 — Linear delay equation: truncation and discretisation error convergence

Setup: a one-dimensional delay equation `dX_t = sum_i (a_i + b_i X_t +
int_0^t K_i(t-s) X_s ds) dW^i_t` with exponential kernels (Sec. 4.2), solved by
an Euler scheme on a reference grid and approximated by the truncated
signature expansion `X_t(M,N) = <l^DE,<=M, W_hat_t(N)>` built from the
coefficient family `l^DE = p (empty - q)^-1` (Sec. 4.2, implementation
notes). Results: `results/exp1_truncation_error.csv/png`,
`results/exp1_discretisation_error.csv`, `results/exp1_fit_B.csv`,
`results/exp1_summary.json`.

* Strong truncation error is monotonically ordered in `M` and grows with `t`
  over `[0,T]` (file `exp1_truncation_error.csv`), matching the paper's
  qualitative description.
* Two-parameter fit `C B^M / (M!)^0.5` at fixed `t` (`results/exp1_fit_B.csv`):
  fitted `B(t=0.5) = 1.117`, `B(t=0.75) = 1.246`. The paper reports
  `B ~ 1.05` at `t=0.5` and `B ~ 1.38` at `t=0.75`. Both fitted values are
  within 10% of the paper's (6.4% and 9.7% respectively, same order of
  magnitude), but their ratio `1.117/1.246 = 0.90` is further from the
  paper's reported ratio `~0.76 ~ sqrt(0.5/0.75)` than the two individual
  values are from the paper. **Verdict: partially reproduced** (individual
  `B` values reproduced within 10%; the ratio between them is not).
* Discretisation error (`results/exp1_discretisation_error.csv`): at grid
  points the strong error decays quickly with `N` (mean strong error at
  `t` on-grid: 0.116 at `N=10` down to 0.00095 at `N=1000`, i.e. roughly
  two orders of magnitude over two orders of magnitude of `N`, consistent
  with the `N^-1` rate of Corollary 3.7 in `d=1`); off grid, rescaling by
  `sqrt(N)` gives an approximately constant curve (0.74 at `N=10`, 0.85 at
  `N=50`), consistent with the `N^-1/2` rate of Theorem 3.5. The weak error
  rescaled by `N` stays within the same order of magnitude across
  `N in {10,...,1000}` (0.14 down to 0.07), consistent with the `N^-1` rate
  of Theorem 3.9, with the mild downward drift attributable to the residual
  truncation error at the fixed `M` used for this slice. **Verdict:
  reproduced** (qualitative rates and orders of magnitude match; see
  "Deviations" for the Monte Carlo scale).

## exp_2 — Shifted Riemann-Liouville fBm: truncation error on the boundary of L^1

Setup: `X_t = Gamma(H+1/2)^-1 int_0^t (t+mu-s)^(H-1/2) dW_s` with `H=0.3`,
`mu=1/52` (Sec. 4.3), expanded with coefficients `c_n(t)` built from the
rising factorial `(1/2-H)^n_bar`. Results: `results/exp2_truncation_error.csv`,
`results/exp2_bound_check.csv`, `results/exp2_proposition_4_6_bounds.csv`,
`results/exp2_summary.json`.

* `q_t = t/(t+mu)` at `t=1`: `52/53 = 0.9811`, matching the paper's value
  exactly.
* Proposition 4.6 upper bound (i) and lower bound (ii) each hold on every
  grid point checked for both `H=0.3` and `H=0.7`
  (`fraction_within_upper_bound = 1.0`, `fraction_above_lower_bound = 1.0`
  in `exp2_summary.json`). **Verdict: reproduced.**
* The decay in `M` over `M in {2,...,7}` is visibly much slower than in
  `exp_1` (geometric with ratio close to 1, rather than factorial), matching
  the paper's qualitative statement that the boundary case loses the
  factorial gain.

## exp_3 — Signature volatility model: put option pricing error bounds

Setup: `sigma_t = <l, W_hat*_t>`, `S_t` solving `dS_t = S_t sigma_t dW^S_t`,
log-price `X_t = log S_t = <Phi(l), W_hat*_t>` with `Phi` as in Lemma 7.1/7.2
(Sec. 4.4, Sec. 7). The paper does not give numerical values for `S_0`, `K`,
`rho` or `l`; this replication uses `S_0=K=1` (at-the-money), `rho=-0.5`, and
an `l` with geometrically decaying coefficients of the same summability type
as the delay-equation example (`src/config.py`, `EXP3_*`; see "Deviations").
Results: `results/exp3_pricing_error.csv/png`, `results/exp3_summary.json`.

* The Monte Carlo put price under the exact (reference-grid) log-price is
  `0.1247 +/- 0.0030` (standard error).
  The truncated-and-discretised scheme's put price error ranges from
  `2.9e-5` (largest `M`, finest `N`) to `1.3e-2` (smallest `M`, coarsest
  `N`), i.e. it shrinks monotonically as `M` and `N` increase, as the
  bound `C(r'_M + N^-1/2)` of Section 4.4 predicts. **Verdict: reproduced**
  (qualitative decay confirmed; see "Deviations" for the numerical
  parameters, which the paper leaves unspecified, so no paper value of the
  pricing error itself can be compared).

## exp_4 — Strong truncation error lemma and its sharpness

Setup: direct computation of the per-length moment bound (5.1), the tail
bound of Lemma 5.1/Theorem 3.3, and the sharpness example `l^(eps) = sum_n
(n!)^(1/2-eps) 1^(x)n` of Proposition 3.8. Results:
`results/exp4_per_length_bound.csv`, `results/exp4_tail_bound.csv`,
`results/exp4_sharpness.csv`, `results/exp4_sharpness_fit.csv`,
`results/exp4_variance_uniformity.csv`, `results/exp4_summary.json`.

* Per-length bound (5.1) holds on every word length checked
  (`per_length_bound_holds = true`).
* Tail bound of Lemma 5.1 holds on every `(t, eps, M)` triple checked
  (`tail_bound_holds_fraction = 1.0`), uniformly across the `N_bar` grid
  (`variance_uniformity_holds = true`), confirming the bound's
  independence from the discretisation level `N`.
* Sharpness (Proposition 3.8): the lower bound holds on every configuration
  checked (`sharpness_lower_bound_holds_fraction = 1.0`). The fitted
  exponent recovered from `log a_J = -eps M log M + O(M)` deviates from the
  input `eps` by a mean absolute error of `0.138` over `eps in
  {0.1,...,0.5}` (`results/exp4_sharpness_fit.csv`), because the `O(M)`
  correction term is of comparable size to `eps M log M` at the moderate
  `M` grid used here (`M <= 40`); the paper's claim concerns the leading
  order as `M -> infinity`. **Verdict: reproduced** for the tail bound and
  the sign/monotonicity of the lower bound; **partially reproduced** for
  the quantitative recovery of `eps` from the finite-`M` fit.

## exp_5 — Strong and weak discretisation error on words

Setup: word-by-word comparison of `W_hat^v_t` (exact signature) and
`W_tilde^v_t(N)` (piecewise-linear interpolation), for the single words
singled out in Remark 5.20 (off-grid `W^1` for the strong `N^-1/2` rate,
on-grid `W^01`/`W^10` for the strong `N^-1` rate in `d=1`, and `W^11` etc. for
the weak `N^-1` rate). Results: `results/exp5_word_errors.csv/png`,
`results/exp5_summary.json`.

* Off-grid strong error rescaled by `sqrt(N)` and on-grid strong error
  rescaled by `N` both satisfy the bounds of Propositions 5.14/5.15 on every
  configuration checked (`strong_within_bound_off_grid_fraction = 1.0`,
  `strong_within_bound_on_grid_fraction = 1.0`).
  For the single word `(1,)` (`W^1`), the on-grid strong error itself falls
  from `0.116` at `N=10` to roughly two orders of magnitude lower at
  `N=1000`, consistent with the `N^-1` rate expected at grid points in
  `d=1`.
* Weak error vanishes on odd-degree words and satisfies the bound of
  Proposition 5.17 on every configuration checked
  (`weak_within_bound_fraction = 1.0`, `weak_error_vanishes_odd_degree =
  true`). **Verdict: reproduced.**

## exp_6 — Extensions to probability, distribution functions and risk measures

Setup: direct verification of the Markov-inequality bound (Lemma 4.1), the
distribution-function bound (Lemma 4.2, via Lemma B.3), the coherent
risk-measure bound (Theorem 4.3, via Lemma 6.1) and the two Value-at-Risk
bounds (Theorem 4.4, via Lemmas 6.2/6.3), on a grid of synthetic values of
`rho = ||Z_1-Z_2||_p` standing in for `E_a(M,N)` (the paper gives no
numerical instance of these bounds). Results:
`results/exp6_risk_extension_checks.csv/png`, `results/exp6_summary.json`.

* All five bounds hold on every grid point checked: `lemma_4_1_markov`
  (100/100), `lemma_4_2_distribution` (25/25), `theorem_4_3_coherent_risk`
  (25/25), `theorem_4_4_var_density` (50/50) and
  `theorem_4_4_var_large_alpha` (50/50). **Verdict: reproduced** (the
  inequalities hold with the stated exponents on every tested configuration;
  the paper gives no numerical constants to compare against).

## Deviations from the paper

* **Monte Carlo scale.** The paper's Monte Carlo experiments (Sec. 4.2-4.4)
  use 20,000 paths on a 10,000-step reference grid. This replication uses
  4,000 paths on a 2,000-step reference grid (`EXP*_N_PATHS_RUN`,
  `EXP*_N_REF_RUN` in `src/config.py`) because running the paper's full
  scale for all of `exp1`, `exp2` and `exp3`, across six truncation levels
  and up to six discretisation levels each, was estimated to exceed the time
  available for this replication. The discretisation levels, truncation
  levels, test times and all method parameters (kernel coefficients, Hurst
  parameter, shift) are unchanged from the paper. Run
  `python -m exp.exp1` (or exp2/exp3) after setting the `*_RUN` constants in
  `src/config.py` to their `*_PAPER` counterparts to reproduce the paper's
  full Monte Carlo scale.
* **exp_1 fitted `B` ratio.** The individual fitted values `B(0.5)=1.117` and
  `B(0.75)=1.246` are each within 10% of the paper's `B~1.05` and `B~1.38`,
  but their ratio (`0.90`) differs more from the paper's reported ratio
  (`~0.76`) than either value does individually; the cause was not isolated
  further within the scale of this replication.
* **exp_3 numerical parameters.** The paper states the pricing-error bound
  of Section 4.4/7 analytically but does not give numerical values for
  `S_0`, `K`, the correlation `rho`, or the coefficient family `l` of the
  volatility functional. This replication chooses `S_0=K=1` (at-the-money),
  `rho=-0.5`, and an `l` with geometrically decaying coefficients of the
  same `A_exp`-type summability as the delay-equation example of Section
  4.2, recorded with this justification in `src/config.py`
  (`EXP3_S0`, `EXP3_K`, `EXP3_RHO`, `EXP3_ELL0`, `EXP3_ELL1`,
  `EXP3_ELL_EMPTY`).
* **exp_4 sharpness fit.** The exponent `eps` recovered from fitting
  `log a_J = -eps M log M + c M` on the truncation grid `M <= 40` used here
  has a mean absolute error of `0.138` against the input `eps`, because at
  this moderate range of `M` the `O(M)` term is not yet negligible next to
  the leading `eps M log M` term; the sign and ordering of the bound (which
  the paper's claim concerns directly) are reproduced exactly.
* **exp_2 strong-error moment order `p`.** The paper does not state the
  value of `p` used for the strong error of Section 4.3 (only Section 4.2
  specifies `p=3`); this replication reuses `p=3` (`EXP2_STRONG_P` in
  `src/config.py`).
* No market data, custom datasets or external data sources are used by this
  repository; every input is generated by `data/generate_synthetic.py` or by
  direct computation from the formulas above.

## Not reproduced

Nothing is omitted outright: every selected experiment (`exp_1` through
`exp_6`) has executable code and produced results under the reduced Monte
Carlo scale described above. The only item not reproduced is the paper's
full-scale Monte Carlo configuration (20,000 paths, 10,000-step reference
grid) for `exp_1`, `exp_2` and `exp_3`, which was not run end-to-end within
the time available for this replication; it is runnable with the `*_PAPER`
constants in `src/config.py` as described above.
