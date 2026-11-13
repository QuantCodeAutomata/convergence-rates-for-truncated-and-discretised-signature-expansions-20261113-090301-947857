"""Experiment 4: strong truncation error lemma and its sharpness
(paper Sec. 3.2, p.10 Proposition 3.8; Sec. 5.1, p.17-21 Lemma 5.1, Theorem 3.3).

Pure closed-form / combinatorial verification -- no Monte Carlo path
simulation is needed here, only the word-counting and moment identities of
``src.error_bounds``.

Run: python -m exp.exp4
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.error_bounds import (
    per_length_claimed_bound,
    per_length_direct_sum,
    per_length_x_t,
    sharpness_lower_rhs,
    sharpness_tail_expectation_log,
    sigma_N_sq,
    tail_bound_closed_form_log,
    tail_bound_direct_log,
)

RESULTS_DIR = Path(config.RESULTS_DIR)


def run() -> dict:
    d = config.EXP4_D
    r = config.EXP4_P
    t_grid = config.EXP4_T_GRID
    eps_grid = config.EXP4_EPS_GRID
    N_bar_grid = config.EXP4_N_BAR_GRID
    M_grid = config.EXP4_M_GRID

    # --- Lemma 5.1 per-length bound (5.1): direct combinatorial sum vs closed form ---
    per_length_rows = []
    for t in t_grid:
        for eps in eps_grid:
            for m in range(0, 25):
                direct = per_length_direct_sum(m, r, t, d, eps)
                claimed = per_length_claimed_bound(m, r, t, d, eps)
                per_length_rows.append({
                    "t": t, "eps": eps, "m": m, "direct_sum": direct, "claimed_bound": claimed,
                    "ratio_direct_over_claimed": direct / claimed if claimed > 0 else np.nan,
                })
    df_per_length = pd.DataFrame(per_length_rows)
    df_per_length.to_csv(RESULTS_DIR / "exp4_per_length_bound.csv", index=False)
    worst_ratio = float(df_per_length["ratio_direct_over_claimed"].max())

    # --- Lemma 5.1 tail bound Delta_p^q: direct tail sum vs closed-form E_eps bound ---
    # Both tails are summed and compared in log-space: the raw per-length
    # bound x_t^m/(m!)^eps keeps growing until m* ~ x_t^{1/eps} (of order
    # 1e3-1e5 here) before the factorial term dominates, so the true tail
    # value is astronomically large long before it is negligible and
    # overflows float64 if exponentiated; only the logarithm is meaningful.
    tail_rows = []
    for t in (0.5, 1.0):
        for eps in (0.3, 0.5):
            for M in (2, 3, 5, 7, 10, 15):
                log_direct = tail_bound_direct_log(M, r, t, d, eps)
                # (M+1+j)! >= (M+1)! j! for M >= 0 (each of the j extra
                # factors M+2,...,M+1+j dominates its counterpart 1,...,j),
                # so re-indexing m = M+1+j in the per-length series gives the
                # exact closed-form bound C_* E_eps(x_t) x_t^{M+1}/((M+1)!)^eps
                # with base x_t itself: B_pq*sqrt(t) = x_t (Lemma 5.1).
                B_pq = per_length_x_t(r, t, d) / np.sqrt(t)
                log_closed = tail_bound_closed_form_log(M, B_pq, t, eps)
                tail_rows.append({
                    "t": t, "eps": eps, "M": M,
                    "log_direct_tail": log_direct, "log_closed_form_tail": log_closed,
                    "direct_minus_closed_log": log_direct - log_closed,
                })
    df_tail = pd.DataFrame(tail_rows)
    df_tail.to_csv(RESULTS_DIR / "exp4_tail_bound.csv", index=False)

    # --- Proposition 3.8 sharpness: ell^{(eps)} = sum_n (n!)^{1/2-eps} 1^{(x)n} ---
    T = 1.0
    sharp_rows = []
    for t in (0.25, 0.5, 0.75, 1.0):
        for eps in (0.1, 0.3, 0.5):
            sigma2 = sigma_N_sq(t, 5000, T)  # reference N, within [t^2/T, t]
            for M in M_grid:
                log_actual = sharpness_tail_expectation_log(M, eps, sigma2)
                log_lower = sharpness_lower_rhs(t, T, eps, M)
                sharp_rows.append({
                    "t": t, "eps": eps, "M": M,
                    "log_actual": log_actual, "log_lower_bound": log_lower,
                    "lower_bound_holds": bool(log_actual >= log_lower - 1e-6),
                })
    df_sharp = pd.DataFrame(sharp_rows)
    df_sharp.to_csv(RESULTS_DIR / "exp4_sharpness.csv", index=False)

    # fit log_actual ~ -eps_fit * M log M + c * M  (recovering the input eps
    # from the empirical decay rate checks that the exponent eps cannot be
    # improved: Theorem 3.3 gives -eps M log M + O(M) as the matching upper bound)
    fit_rows = []
    for (t, eps), sub in df_sharp.groupby(["t", "eps"]):
        sub = sub[sub["M"] >= 6].sort_values("M")
        M = sub["M"].to_numpy(dtype=float)
        y = sub["log_actual"].to_numpy()
        X = np.vstack([-(M * np.log(M)), M]).T
        (eps_fit, c_fit), *_ = np.linalg.lstsq(X, y, rcond=None)
        fit_rows.append({"t": t, "eps_input": eps, "eps_fit": float(eps_fit), "c_fit": float(c_fit)})
    df_fit = pd.DataFrame(fit_rows)
    df_fit.to_csv(RESULTS_DIR / "exp4_sharpness_fit.csv", index=False)

    # --- uniformity in N of sigma_N(t)^2 in [t^2/T, t] ---
    unif_rows = []
    for t in t_grid:
        for N in N_bar_grid:
            s2 = sigma_N_sq(t, N, T)
            unif_rows.append({
                "t": t, "N": N, "sigma_N_sq": s2,
                "within_bounds": bool(t ** 2 / T - 1e-9 <= s2 <= t + 1e-9),
            })
    df_unif = pd.DataFrame(unif_rows)
    df_unif.to_csv(RESULTS_DIR / "exp4_variance_uniformity.csv", index=False)

    _plot(df_per_length, df_sharp, df_fit)

    summary = {
        "worst_ratio_direct_over_claimed_bound": worst_ratio,
        "per_length_bound_holds": bool(worst_ratio <= 1.0 + 1e-6),
        "tail_bound_holds_fraction": float((df_tail["direct_minus_closed_log"] <= 1e-6).mean()),
        "sharpness_lower_bound_holds_fraction": float(df_sharp["lower_bound_holds"].mean()),
        "eps_fit_mean_abs_error": float(np.mean(np.abs(df_fit["eps_fit"] - df_fit["eps_input"]))),
        "variance_uniformity_holds": bool(df_unif["within_bounds"].all()),
    }
    with open(RESULTS_DIR / "exp4_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp4] {summary}")
    return summary


def _plot(df_per_length: pd.DataFrame, df_sharp: pd.DataFrame, df_fit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sub = df_per_length[(np.isclose(df_per_length["t"], 0.5)) & (df_per_length["eps"] == 0.5)]
    axes[0].semilogy(sub["m"], sub["direct_sum"], "o-", label="direct combinatorial sum")
    axes[0].semilogy(sub["m"], sub["claimed_bound"], "--", label="C* x_t^m/(m!)^eps")
    axes[0].set_xlabel("m")
    axes[0].set_title("Per-length bound (5.1), t=0.5, eps=0.5")
    axes[0].legend(fontsize=7)

    for (t, eps), g in df_sharp.groupby(["t", "eps"]):
        if not np.isclose(t, 1.0):
            continue
        g = g.sort_values("M")
        axes[1].plot(g["M"], g["log_actual"], "o-", ms=3, label=f"eps={eps}")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("log E[X(inf,N)-X(M,N)]")
    axes[1].set_title("Sharpness example decay, t=1")
    axes[1].legend(fontsize=7)

    axes[2].scatter(df_fit["eps_input"], df_fit["eps_fit"])
    lims = [0.0, 0.55]
    axes[2].plot(lims, lims, "k--", lw=1)
    axes[2].set_xlabel("input eps")
    axes[2].set_ylabel("fitted exponent from -eps M logM slope")
    axes[2].set_title("Sharpness: recovered exponent")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_verification.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    run()
