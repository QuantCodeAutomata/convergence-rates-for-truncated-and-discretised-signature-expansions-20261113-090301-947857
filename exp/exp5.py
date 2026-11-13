"""Experiment 5: strong and weak discretisation error on words
(paper Sec. 5.2, p.22-23, Propositions 5.14/5.15; Sec. 5.3, p.25-26,
Proposition 5.17).

Single words v of Brownian motion's signature are compared between the fine
reference grid (``N_ref`` steps, a numerical proxy for the exact/Stratonovich
signature ``Ŵ^v``) and coarser N-step grids of the *same* path
(``W̃^v(N)``), at a fixed off-grid point and at the grid point t=T.

Run: python -m exp.exp5 [--full]
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

from data.generate_synthetic import aggregate_increments, brownian_increments, checkpoint_fractions, time_extended_deltas
from src import config
from src.discretisation_bounds import strong_bound_rhs, strong_bound_rhs_grid, weak_bound_rhs, weighted_length
from src.path_signatures import levels_along_path

RESULTS_DIR = Path(config.RESULTS_DIR)
L_ALPHABET = 2  # letters {0 (time), 1 (single Brownian direction)}, Sec. 5.2


def _word_component(levels, word) -> np.ndarray:
    n = len(word)
    return levels[n][(slice(None),) + word]


def run(full: bool) -> dict:
    N_ref = config.EXP5_N_REF_PAPER if full else config.EXP5_N_REF_RUN
    n_paths = config.EXP5_N_PATHS_PAPER if full else config.EXP5_N_PATHS_RUN
    N_levels = [N for N in config.EXP5_N_LEVELS if N_ref % N == 0]
    T = config.EXP5_T
    d = config.EXP5_D
    p = config.EXP5_STRONG_P
    words_strong = config.EXP5_WORDS_STRONG
    words_weak = config.EXP5_WORDS_WEAK
    M_max = max(len(w) for w in words_strong + words_weak)
    off_grid_fraction = 0.37

    print(f"[exp5] N_ref={N_ref} n_paths={n_paths} N_levels={N_levels} words_strong={words_strong} words_weak={words_weak}")

    dW_ref = brownian_increments(n_paths, N_ref, T, dim=1, seed=config.RNG_SEED + 3)
    deltas_ref = time_extended_deltas(dW_ref, T)

    rows = []
    for N in N_levels:
        dW_N = aggregate_increments(dW_ref, N_ref, N)
        deltas_N = time_extended_deltas(dW_N, T)

        k_off = N // 3
        t_off = (k_off + off_grid_fraction) / N * T
        key_on_N, key_off_N = (N - 1, 1.0), (k_off, off_grid_fraction)
        key_on_ref = (N_ref - 1, 1.0)
        key_off_ref = checkpoint_fractions((t_off,), N_ref, T)[0]

        levels_N = levels_along_path(deltas_N, M_max, L_ALPHABET, [key_on_N, key_off_N])
        levels_ref = levels_along_path(deltas_ref, M_max, L_ALPHABET, [key_on_ref, key_off_ref])

        for word in set(words_strong) | set(words_weak):
            w_len = weighted_length(word)
            W_hat_on = _word_component(levels_ref[key_on_ref], word)
            W_tilde_on = _word_component(levels_N[key_on_N], word)
            W_hat_off = _word_component(levels_ref[key_off_ref], word)
            W_tilde_off = _word_component(levels_N[key_off_N], word)

            diff_on, diff_off = W_hat_on - W_tilde_on, W_hat_off - W_tilde_off
            strong_on = float(np.mean(np.abs(diff_on) ** p) ** (1.0 / p))
            strong_off = float(np.mean(np.abs(diff_off) ** p) ** (1.0 / p))
            weak_on = float(abs(np.mean(diff_on)))
            weak_off = float(abs(np.mean(diff_off)))

            rows.append({
                "N": N, "word": str(word), "weighted_length": w_len,
                "strong_error_off_grid": strong_off,
                "strong_error_off_grid_x_sqrtN": strong_off * np.sqrt(N),
                "strong_bound_off_grid": strong_bound_rhs(word, T, N),
                "strong_error_on_grid": strong_on,
                "strong_error_on_grid_x_N": strong_on * N,
                "strong_bound_on_grid_d1": strong_bound_rhs_grid(word, T, N),
                "weak_error_off_grid": weak_off,
                "weak_error_off_grid_x_N": weak_off * N,
                "weak_error_on_grid": weak_on,
                "weak_error_on_grid_x_N": weak_on * N,
                "weak_bound": weak_bound_rhs(word, T, N, d),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp5_word_errors.csv", index=False)

    _plot(df)

    summary = {
        "N_ref": N_ref, "n_paths": n_paths, "N_levels": N_levels,
        "words_strong": [str(w) for w in words_strong], "words_weak": [str(w) for w in words_weak],
        "strong_within_bound_off_grid_fraction": float((df["strong_error_off_grid"] <= 5 * df["strong_bound_off_grid"]).mean()),
        "strong_within_bound_on_grid_fraction": float((df["strong_error_on_grid"] <= 5 * df["strong_bound_on_grid_d1"]).mean()),
        "weak_within_bound_fraction": float((df["weak_error_off_grid"] <= 5 * df["weak_bound"] + 1e-10).mean()),
        "weak_error_vanishes_odd_degree": bool(
            df.loc[df["weighted_length"] % 2 == 1, "weak_error_off_grid"].abs().max() < 1e-2
        ),
    }
    with open(RESULTS_DIR / "exp5_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp5] {summary}")
    return summary


def _plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for word, sub in df.groupby("word"):
        sub = sub.sort_values("N")
        axes[0].loglog(sub["N"], sub["strong_error_off_grid"], "o-", label=word)
        axes[1].semilogx(sub["N"], sub["strong_error_on_grid_x_N"], "o-", label=word)
        axes[2].semilogx(sub["N"], sub["weak_error_off_grid_x_N"], "o-", label=word)
    axes[0].set_title("Off-grid strong error ||.||_p (rate N^-1/2)")
    axes[1].set_title("On-grid strong error x N (d=1, rate N^-1)")
    axes[2].set_title("Weak error x N (rate N^-1)")
    for ax in axes:
        ax.set_xlabel("N")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_word_errors.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's N_ref/n_paths")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    run(args.full)
