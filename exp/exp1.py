"""Experiment 1: linear delay equation -- truncation and discretisation error
(paper Sec. 4.2, p.13; Theorems 3.3, 3.5, 3.9, Corollaries 3.6, 3.7, 3.11).

Run: python -m exp.exp1 [--full]
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

from data.generate_synthetic import (
    aggregate_increments,
    brownian_increments,
    checkpoint_fractions,
    time_extended_deltas,
)
from src import config
from src import tensor_algebra as ta
from src.delay_equation import build_ell_DE, simulate_reference
from src.path_signatures import levels_along_path, truncated_inner_product

RESULTS_DIR = Path(config.RESULTS_DIR)
L_ALPHABET = 2  # letters {0 (time), 1 (single Brownian direction)}


def run(full: bool) -> dict:
    N_ref = config.EXP1_N_REF_PAPER if full else config.EXP1_N_REF_RUN
    n_paths = config.EXP1_N_PATHS_PAPER if full else config.EXP1_N_PATHS_RUN
    M_levels = config.EXP1_M_LEVELS
    N_levels = [N for N in config.EXP1_N_LEVELS if N_ref % N == 0]
    test_times = config.EXP1_TEST_TIMES
    T = config.EXP1_T
    p = config.EXP1_STRONG_P
    M_max = max(M_levels)

    print(f"[exp1] N_ref={N_ref} n_paths={n_paths} M_levels={M_levels} N_levels={N_levels}")

    ell = build_ell_DE(
        config.EXP1_X0, config.EXP1_A, config.EXP1_B, config.EXP1_ALPHA, config.EXP1_BETA, M_max
    )

    dW_ref = brownian_increments(n_paths, N_ref, T, dim=1, seed=config.RNG_SEED)
    X_ref_path = simulate_reference(
        dW_ref, config.EXP1_X0, config.EXP1_A, config.EXP1_B, config.EXP1_ALPHA, config.EXP1_BETA, T
    )

    deltas_ref = time_extended_deltas(dW_ref, T)
    frac_ref = checkpoint_fractions(test_times, N_ref, T)
    levels_ref = levels_along_path(deltas_ref, M_max, L_ALPHABET, frac_ref)

    # reference value at each test time (Euler solution, independent of M, N)
    X_ref_at_t = {}
    for t, (k, f) in zip(test_times, frac_ref):
        X_ref_at_t[t] = X_ref_path[:, k + 1] if f >= 1.0 - 1e-9 else _interp(X_ref_path, k, f)

    # --- truncation error: X_t(M, N_ref) vs reference, uniform in N (use N_ref) ---
    rows = []
    for M in M_levels:
        for t, key in zip(test_times, frac_ref):
            X_M = truncated_inner_product(ell, levels_ref[key], M)
            diff = X_ref_at_t[t] - X_M
            strong_err = float(np.mean(np.abs(diff) ** p) ** (1.0 / p))
            weak_err = float(abs(np.mean(diff)))
            rows.append({"M": M, "t": t, "strong_error": strong_err, "weak_error": weak_err})
    df_trunc = pd.DataFrame(rows)
    df_trunc.to_csv(RESULTS_DIR / "exp1_truncation_error.csv", index=False)

    # --- fit C B^M / (M!)^{1/2} at fixed times ---
    fit_rows = []
    for t_fit in config.EXP1_FIT_TIMES:
        sub = df_trunc[np.isclose(df_trunc["t"], t_fit)].sort_values("M")
        Ms = sub["M"].to_numpy(dtype=float)
        errs = sub["strong_error"].to_numpy()
        valid = errs > 0
        y = np.log(errs[valid]) + 0.5 * np.array([np.sum(np.log(np.arange(1, m + 1))) for m in Ms[valid]])
        A = np.vstack([Ms[valid], np.ones_like(Ms[valid])]).T
        (log_B, log_C), *_ = np.linalg.lstsq(A, y, rcond=None)
        fit_rows.append({"t": t_fit, "B": float(np.exp(log_B)), "C": float(np.exp(log_C))})
    df_fit = pd.DataFrame(fit_rows)
    df_fit.to_csv(RESULTS_DIR / "exp1_fit_B.csv", index=False)

    # --- discretisation error: dense off-grid + on-grid times, M fixed ---
    dense_times = np.round(np.linspace(0.0, 1.0, 101), 6)
    disc_rows = []
    for N in N_levels:
        dW_N = aggregate_increments(dW_ref, N_ref, N)
        deltas_N = time_extended_deltas(dW_N, T)
        frac_N = checkpoint_fractions(tuple(dense_times), N, T)
        levels_N = levels_along_path(deltas_N, M_max, L_ALPHABET, frac_N)
        frac_ref_dense = checkpoint_fractions(tuple(dense_times), N_ref, T)
        for M in config.EXP1_DISC_M_FIXED:
            for t, key_N, key_ref in zip(dense_times, frac_N, frac_ref_dense):
                X_N = truncated_inner_product(ell, levels_N[key_N], M)
                X_ref_full = truncated_inner_product(ell, levels_ref[key_ref], M) if key_ref in levels_ref \
                    else _nearest_ref(levels_ref, frac_ref, ell, M, t)
                diff = X_ref_full - X_N
                strong_err = float(np.mean(np.abs(diff) ** p) ** (1.0 / p))
                weak_err = float(abs(np.mean(diff)))
                is_grid = key_N[1] >= 1.0 - 1e-9
                disc_rows.append({
                    "N": N, "M": M, "t": float(t), "on_grid": is_grid,
                    "strong_error": strong_err, "weak_error": weak_err,
                    "strong_error_x_sqrtN": strong_err * np.sqrt(N),
                    "weak_error_x_N": weak_err * N,
                })
    df_disc = pd.DataFrame(disc_rows)
    df_disc.to_csv(RESULTS_DIR / "exp1_discretisation_error.csv", index=False)

    _plot_truncation(df_trunc)
    _plot_discretisation(df_disc)

    summary = {
        "N_ref": N_ref, "n_paths": n_paths, "M_levels": list(M_levels), "N_levels": N_levels,
        "fit_B": fit_rows,
        "max_strong_truncation_error": float(df_trunc["strong_error"].max()),
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp1] fitted B: {fit_rows}")
    return summary


def _interp(X_path: np.ndarray, k: int, f: float) -> np.ndarray:
    return (1 - f) * X_path[:, k] + f * X_path[:, k + 1]


def _nearest_ref(levels_ref, frac_ref, ell, M, t):
    idx = int(np.argmin([abs(tt - t) for tt in frac_ref]))
    key = frac_ref[idx]
    return truncated_inner_product(ell, levels_ref[key], M)


def _plot_truncation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for M, sub in df.groupby("M"):
        sub = sub.sort_values("t")
        axes[0].plot(sub["t"], sub["strong_error"], marker="o", ms=3, label=f"M={M}")
        axes[1].plot(sub["t"], sub["weak_error"], marker="o", ms=3, label=f"M={M}")
    axes[0].set_title("Strong truncation error ||X_t - X_t(M,N_ref)||_p")
    axes[1].set_title("Weak truncation error |E[X_t - X_t(M,N_ref)]|")
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_truncation_error.png", dpi=140)
    plt.close(fig)


def _plot_discretisation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (N, M), sub in df.groupby(["N", "M"]):
        sub = sub.sort_values("t")
        axes[0].plot(sub["t"], sub["strong_error_x_sqrtN"], lw=1, label=f"N={N},M={M}")
        axes[1].plot(sub["t"], sub["weak_error_x_N"], lw=1, label=f"N={N},M={M}")
    axes[0].set_title("Strong discretisation error x sqrt(N) (scalloped, Prop. 5.14/5.15)")
    axes[1].set_title("Weak discretisation error x N (flat, Prop. 5.17)")
    for ax in axes:
        ax.set_xlabel("t")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_discretisation_error.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's N_ref/n_paths")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    run(args.full)
