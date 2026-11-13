# Convergence rates for truncated and discretised signature expansions

Replication of selected numerical experiments and theoretical verifications from

> *Convergence rates for truncated and discretised signature expansions* (arXiv:2611.16087).

The paper studies linear functionals of the (Stratonovich) signature of a path
`(t, W_t)` built from a Brownian motion `W`, and bounds two sources of error when
such a functional is used to approximate a stochastic process: the **truncation
error** (keeping only signature levels up to `M`) and the **discretisation
error** (replacing the continuous path by its piecewise-linear interpolation on
`N` grid points). This repository reproduces the paper's numerical section
(Section 4) and its supporting theoretical lemmas (Sections 3, 5, 6, 7) through
six experiments.

No market data is used: every experiment simulates Brownian paths or verifies
purely analytic inequalities, as specified in the paper's Section 4.

## Experiments

| Experiment | Title | Paper section |
|---|---|---|
| `exp1` | Linear delay equation: truncation and discretisation error convergence | Sec. 4.2 |
| `exp2` | Shifted Riemann-Liouville fractional Brownian motion: truncation error on the boundary of `L^1` | Sec. 4.3 |
| `exp3` | Signature volatility model: put option pricing error bounds | Sec. 4.4, 7 |
| `exp4` | Strong truncation error lemma and its sharpness | Sec. 3.2, 5.1 |
| `exp5` | Strong and weak discretisation error on words | Sec. 5.2, 5.3 |
| `exp6` | Extensions to probability, distribution functions and risk measures | Sec. 4.1, 6 |

## Repository layout

```
src/     signature machinery, delay-equation and fBm coefficient families,
         discretisation/truncation error bounds, risk-measure extensions,
         experiment parameters (src/config.py)
exp/     one script per experiment, exp1..exp6, calling only src/
data/    data/generate_synthetic.py generates the Brownian paths used by
         every experiment (no market data is used by this paper)
tests/   unit tests for the shared src/ modules
results/ CSV/JSON/PNG outputs of the experiments and RESULTS.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest -q
```

## Running the experiments

Each experiment is runnable as a module from the repository root:

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
```

Every run (re)writes its artifacts directly into `results/`. The default
configuration in `src/config.py` uses a Monte Carlo sample of 4,000 paths on a
2,000-step reference grid (`*_RUN` constants) instead of the paper's 20,000
paths on a 10,000-step grid (`*_PAPER` constants), for the reason recorded in
`results/RESULTS.md`. To run an experiment at the paper's full scale, edit
`src/config.py` and use the `*_PAPER` values in place of the `*_RUN` values
(for example set `EXP1_N_REF_RUN = EXP1_N_REF_PAPER` and
`EXP1_N_PATHS_RUN = EXP1_N_PATHS_PAPER` before calling `python -m exp.exp1`).

## Reproducing everything

```bash
./reproduce.sh
```

runs the test suite and all six experiments in sequence, overwriting the
contents of `results/`.

See `results/RESULTS.md` for the comparison against the paper's reported
values, the deviations introduced by this implementation, and the results the
paper reports that are not reproduced here.
