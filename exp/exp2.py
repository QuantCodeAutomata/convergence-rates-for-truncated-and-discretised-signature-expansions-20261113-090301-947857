"""Experiment 2: shifted Riemann-Liouville fractional Brownian motion --
truncation error on the boundary of L^1 (paper Sec. 4.3, p.15, Proposition 4.6).

Run: python -m exp.exp2 [--full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import brownian_increments, checkpoint_fractions, time_extended_deltas
from src import config
from src.fbm import build_ell_RL, geometric_ratio, proposition_4_6_lower, proposition_4_6_upper
from src.path_signatures import levels_along_path, truncated_inner_product

RESULTS_DIR = Path(config.RESULTS_DIR)
L_ALPHABET = 2  # letters {0 (time), 1 (single Brownian direction)}


def run(full: bool) -> dict:
    N_ref = config.EXP2_N_REF_PAPER if full else config.EXP2_N_REF_RUN
    n_paths = config.EXP2_N_PATHS_PAPER if full else config.EXP2_N_PATHS_RUN
    M_levels = config.EXP2_M_LEVELS
    M_ref = config.EXP2_M_REF
    test_times = [t for t in config.EXP2_TEST_TIMES if t > 0.0]
    T = config.EXP2_T
    p = config.EXP2_STRONG_P
    M_max = max(M_levels + (M_ref,))

    print(f"[exp2] N_ref={N_ref} n_paths={n_paths} M_levels={M_levels} M_ref={M_ref}")

    dW_ref = brownian_increments(n_paths, N_ref, T, dim=1, seed=config.RNG_SEED + 1)
    deltas_ref = time_extended_deltas(dW_ref, T)
    frac_ref = checkpoint_fractions(tuple(test_times), N_ref, T)

    rows, bound_rows = [], []
    for H_label, H in (("H=0.3", config.EXP2_H), ("H=0.7", config.EXP2_H_ALT)):
        for t, key in zip(test_times, frac_ref):
            ell_full = build_ell_RL(t, H, config.EXP2_MU, M_max)
            levels = levels_along_path(deltas_ref, M_max, L_ALPHABET, [key])[key]
            X_ref = truncated_inner_product(ell_full, levels, M_ref)
            for M in M_levels:
                X_M = truncated_inner_product(ell_full, levels, M)
                diff = X_ref - X_M
                strong_err = float(np.mean(np.abs(diff) ** p) ** (1.0 / p))
                weak_err = float(np.mean(np.cos(X_ref) - np.cos(X_M)))
                rows.append({
                    "H": H_label, "t": t, "M": M,
                    "strong_error": strong_err, "weak_error_cos": abs(weak_err),
                })
                q_t = geometric_ratio(t, config.EXP2_MU)
                bound_rows.append({
                    "H": H_label, "t": t, "M": M, "q_t": q_t,
                    "upper_bound": proposition_4_6_upper(t, M, H, config.EXP2_MU, p),
                    "lower_bound": proposition_4_6_lower(t, M, H, config.EXP2_MU),
                    "strong_error": strong_err,
                })
    df = pd.DataFrame(rows)
    df_bounds = pd.DataFrame(bound_rows)
    df.to_csv(RESULTS_DIR / "exp2_truncation_error.csv", index=False)
    df_bounds.to_csv(RESULTS_DIR / "exp2_proposition_4_6_bounds.csv", index=False)

    # check: for H != 1/2 the lower bound <= empirical strong error <= upper bound
    sat = df_bounds.copy()
    sat["satisfies_upper"] = sat["strong_error"] <= 3.0 * sat["upper_bound"] + 1e-8
    sat["satisfies_lower"] = sat["strong_error"] >= sat["lower_bound"] / 3.0 - 1e-8
    sat.to_csv(RESULTS_DIR / "exp2_bound_check.csv", index=False)

    _plot(df, df_bounds)

    summary = {
        "N_ref": N_ref, "n_paths": n_paths, "M_levels": list(M_levels), "M_ref": M_ref,
        "q_t_at_1": geometric_ratio(1.0, config.EXP2_MU),
        "fraction_within_upper_bound(x3)": float(sat["satisfies_upper"].mean()),
        "fraction_above_lower_bound(/3)": float(sat["satisfies_lower"].mean()),
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp2] q_t(T=1) = {summary['q_t_at_1']:.4f}")
    return summary


def _plot(df: pd.DataFrame, df_bounds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (H_label, M), sub in df.groupby(["H", "M"]):
        if H_label != "H=0.3":
            continue
        sub = sub.sort_values("t")
        axes[0].plot(sub["t"], sub["strong_error"], marker="o", ms=3, label=f"M={M}")
        axes[1].plot(sub["t"], sub["weak_error_cos"], marker="o", ms=3, label=f"M={M}")
    axes[0].set_title("Strong truncation error ||X_t(inf)-X_t(M)||_p, H=0.3")
    axes[1].set_title("Weak error |E[cos X_t(inf) - cos X_t(M)]|, H=0.3")
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_truncation_error.png", dpi=140)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    sub = df_bounds[(df_bounds["H"] == "H=0.3") & np.isclose(df_bounds["t"], 1.0)].sort_values("M")
    ax.semilogy(sub["M"], sub["strong_error"], "o-", label="empirical")
    ax.semilogy(sub["M"], sub["upper_bound"], "--", label="Prop. 4.6(i) upper bound")
    ax.semilogy(sub["M"], sub["lower_bound"], ":", label="Prop. 4.6(ii) lower bound")
    ax.set_xlabel("M")
    ax.set_title("Geometric truncation rate at t=1, H=0.3")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "exp2_bounds_at_t1.png", dpi=140)
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's N_ref/n_paths")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    run(args.full)
