"""Experiment 3: signature volatility model -- put option pricing error bounds
(paper Sec. 4.4, p.16, and Sec. 7 / Proposition 7.3, p.28).

Run: python -m exp.exp3 [--full]
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
from src.path_signatures import levels_along_path, truncated_inner_product
from src.sig_volatility import build_ell_vol, phi_map, put_price_mc, simulate_log_price_reference

RESULTS_DIR = Path(config.RESULTS_DIR)
L_ALPHABET = 3  # letters {0 (time), 1 (W), 2 (W^perp)}


def _time_extend_two_dim(dW: np.ndarray, dW_perp: np.ndarray, T: float) -> np.ndarray:
    n_paths, N, _ = dW.shape
    h = T / N
    time_col = np.full((n_paths, N, 1), h)
    return np.concatenate([time_col, dW, dW_perp], axis=2)


def run(full: bool) -> dict:
    N_ref = config.EXP3_N_REF_PAPER if full else config.EXP3_N_REF_RUN
    n_paths = config.EXP3_N_PATHS_PAPER if full else config.EXP3_N_PATHS_RUN
    M_levels = config.EXP3_M_LEVELS
    N_levels = [N for N in config.EXP3_N_LEVELS if N_ref % N == 0]
    T = config.EXP3_T
    log_s0 = np.log(config.EXP3_S0)
    M_max = max(M_levels)

    print(f"[exp3] N_ref={N_ref} n_paths={n_paths} M_levels={M_levels} N_levels={N_levels}")

    ell = build_ell_vol(config.EXP3_ELL_EMPTY, config.EXP3_ELL0, config.EXP3_ELL1)
    ell_phi = phi_map(ell, config.EXP3_RHO, log_s0, M_max)

    rng_seed = config.RNG_SEED + 2
    dW = brownian_increments(n_paths, N_ref, T, dim=1, seed=rng_seed)
    dW_perp = brownian_increments(n_paths, N_ref, T, dim=1, seed=rng_seed + 1)
    log_price_ref_path = simulate_log_price_reference(
        dW, dW_perp, config.EXP3_ELL_EMPTY, config.EXP3_ELL0, config.EXP3_ELL1,
        config.EXP3_RHO, log_s0, T,
    )
    X_ref_T = log_price_ref_path[:, -1]
    price_exact, se_exact = put_price_mc(X_ref_T, config.EXP3_K)

    deltas_ref = _time_extend_two_dim(dW, dW_perp, T)
    key_T = checkpoint_fractions((T,), N_ref, T)[0]
    levels_ref_T = levels_along_path(deltas_ref, M_max, L_ALPHABET, [key_T])[key_T]

    rows = []
    for M in M_levels:
        # X_t(M, N_ref) = <Phi(ell)^{<=M}, W-hat*_t(N_ref)> = <Phi(ell^{<=M})^{<=M}, ...> (Lemma 7.2)
        X_M_ref = truncated_inner_product(ell_phi, levels_ref_T, M)
        price_M_ref, se_M_ref = put_price_mc(X_M_ref, config.EXP3_K)
        strong_err = float(np.mean(np.abs(X_ref_T - X_M_ref) ** 2) ** 0.5)
        pricing_err = abs(price_exact - price_M_ref)
        for N in N_levels:
            dW_N = dW.reshape(n_paths, N, N_ref // N, 1).sum(axis=2)
            dW_perp_N = dW_perp.reshape(n_paths, N, N_ref // N, 1).sum(axis=2)
            deltas_N = _time_extend_two_dim(dW_N, dW_perp_N, T)
            key_T_N = checkpoint_fractions((T,), N, T)[0]
            levels_N_T = levels_along_path(deltas_N, M_max, L_ALPHABET, [key_T_N])[key_T_N]
            X_MN = truncated_inner_product(ell_phi, levels_N_T, M)
            price_MN, se_MN = put_price_mc(X_MN, config.EXP3_K)
            rows.append({
                "M": M, "N": N,
                "strong_error_vs_ref_N": float(np.mean(np.abs(X_ref_T - X_MN) ** 2) ** 0.5),
                "put_price_MN": price_MN, "put_price_se": se_MN,
                "pricing_error_abs": abs(price_exact - price_MN),
            })
        rows.append({
            "M": M, "N": N_ref,
            "strong_error_vs_ref_N": strong_err,
            "put_price_MN": price_M_ref, "put_price_se": se_M_ref,
            "pricing_error_abs": pricing_err,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp3_pricing_error.csv", index=False)

    _plot(df)

    summary = {
        "N_ref": N_ref, "n_paths": n_paths, "M_levels": list(M_levels), "N_levels": N_levels,
        "exact_put_price": price_exact, "exact_put_price_se": se_exact,
        "min_pricing_error_abs": float(df["pricing_error_abs"].min()),
        "max_pricing_error_abs": float(df["pricing_error_abs"].max()),
    }
    with open(RESULTS_DIR / "exp3_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp3] exact put price = {price_exact:.5f} +- {se_exact:.5f}")
    return summary


def _plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for M, sub in df.groupby("M"):
        sub = sub.sort_values("N")
        axes[0].plot(sub["N"], sub["strong_error_vs_ref_N"], marker="o", label=f"M={M}")
        axes[1].plot(sub["N"], sub["pricing_error_abs"], marker="o", label=f"M={M}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_title("Strong error ||X_T - X_T(M,N)||_2")
    axes[1].set_title("|E[Pi(S_T)] - E[Pi(exp(X_T(M,N)))]|")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_pricing_error.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's N_ref/n_paths")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    run(args.full)
