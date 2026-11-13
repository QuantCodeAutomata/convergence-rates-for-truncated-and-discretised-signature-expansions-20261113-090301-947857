# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A replication of six numerical/theoretical experiments from *Convergence
rates for truncated and discretised signature expansions*: truncation and
discretisation error convergence for a linear delay equation (`exp1`), the
geometric truncation rate for a shifted Riemann-Liouville fractional Brownian
motion (`exp2`), pricing error bounds for a European put under a signature
volatility model (`exp3`), the strong truncation error lemma and its
sharpness (`exp4`), word-by-word strong/weak discretisation error bounds
(`exp5`), and the extensions to probability, distribution-function and risk-
measure error bounds (`exp6`). See `README.md` for the paper-section mapping
and `results/RESULTS.md` for the numerical comparison against the paper.

## Layout

```
src/     flat library modules: tensor algebra on words/signatures
         (src/tensor_algebra.py, src/path_signatures.py), the delay-equation
         and fBm coefficient families (src/delay_equation.py, src/fbm.py),
         the signature volatility model (src/sig_volatility.py), truncation/
         discretisation error bounds (src/error_bounds.py,
         src/discretisation_bounds.py), the probability/risk-measure
         extensions (src/risk_extensions.py), and all experiment parameters
         in src/config.py (module-level UPPER_CASE variables only)
exp/     one script per experiment (exp1..exp6); each only calls src/ and
         data/, never re-implements a shared component
data/    data/generate_synthetic.py simulates every Brownian path used by the
         experiments (this paper uses no market data, so there is no
         data_loader.py)
tests/   pytest unit tests for the src/ modules, run on small/synthetic inputs
results/ CSV/JSON/PNG artifacts written by the experiments, plus RESULTS.md
```

## Commands

Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -q
```

(there are no `market_data`-marked tests in this repository since no market
data is used; `python -m pytest -m "not market_data"` is a no-op equivalent.)

Experiments:

```bash
python -m exp.exp1   # ... through exp6
```

or `./reproduce.sh` to run the tests and all six experiments in sequence.

Running an experiment at the paper's full Monte Carlo scale (20,000 paths on
a 10,000-step reference grid) requires editing `src/config.py` to use the
`*_PAPER` constants instead of the `*_RUN` constants used by default; see
`results/RESULTS.md` for the reason the default run uses a smaller scale and
the resulting estimated runtime of the full configuration.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables, each commented with its meaning, unit and the
  paper section it comes from (or "not specified in the paper").
- Data preparation code lives only in `data/`; `src/` and `exp/` must not
  generate paths or load data directly.
- `exp/*.py` call the public functions of `src/` with their documented
  signatures; they do not invent private helpers that duplicate a shared
  component (signature truncation, error-bound formulas, path generation).
- Tests live only in `tests/` and run on small, fast synthetic inputs; no
  test depends on a network call.
- No API keys are used by this repository (no market data). If a future
  experiment ever needed one, it would be read from an uncommitted `.env`
  file, never hardcoded.
- After rerunning the experiments, the files in `results/` are overwritten
  by the new run. The prose, numbers and verdicts in `results/RESULTS.md`
  are not regenerated automatically: check them against the new result
  files and update them by hand.
