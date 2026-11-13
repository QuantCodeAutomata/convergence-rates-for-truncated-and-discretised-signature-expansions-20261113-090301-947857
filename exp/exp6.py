"""Experiment 6: extensions to probability, distribution functions and risk
measures (paper Sec. 4.1, p.11-12; proofs in Sec. 6, p.26-27).

Each extension bound of Sec. 4.1 is applied to an explicit pair of Gaussian
random variables Z_2 ~ N(0,1) (standing for g(X_t)) and Z_1 = Z_2 + D (standing
for g(X_t(M,N))), where D is an independent centred Gaussian chosen so that
its p-th absolute moment ||D||_p equals a prescribed value rho = C E_a(M,N)
(``src.fbm.kappa_p`` gives the exact Gaussian absolute-moment constant). This
keeps the genuinely signature-specific error (verified in the other
experiments) separate from the real-analysis chain of inequalities checked
here (Markov, Lemma B.3, Hoelder/Lemma 6.1, quantile stability/Lemma 6.3),
exactly as ``src.risk_extensions`` isolates it.

VaR/ES use the standard upper-tail quantile convention VaR_alpha(Z) =
F_Z^{-1}(alpha), ES_alpha(Z) = E[Z | Z >= VaR_alpha(Z)], consistent with
Lemma 6.2's ordering VaR_alpha <= ES_alpha and with the high confidence
levels alpha in {0.95, 0.99} of Theorem 4.4.

Run: python -m exp.exp6
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from src import config
from src.fbm import kappa_p
from src.risk_extensions import (
    lemma_4_1_probability_bound,
    lemma_4_2_distribution_bound,
    lemma_6_2_var_leq_es,
    lemma_6_3_quantile_stability,
    theorem_4_3_coherent_risk_bound,
    theorem_4_4_var_bound_large_alpha,
)

RESULTS_DIR = Path(config.RESULTS_DIR)


def _gaussian_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    z_alpha = norm.ppf(alpha)
    var = mu + sigma * z_alpha
    es = mu + sigma * norm.pdf(z_alpha) / (1.0 - alpha)
    return var, es


def run() -> dict:
    p = config.EXP6_P
    rho_grid = config.EXP6_RHO_GRID
    eta0, f_sup, J_half, I_half = 0.5, norm.pdf(0.0), 2.0, 3.0  # Lemma 4.2 setup, not specified in the paper
    theta = config.EXP6_THETA
    q = config.EXP6_Q_RISK
    q_star = q / (q - 1.0)
    C_rho = config.EXP6_C_RHO
    c_density, gamma = config.EXP6_DENSITY_C, config.EXP6_GAMMA

    rows = []
    for rho in rho_grid:
        sigma_D = rho / kappa_p(p)  # ||D||_p = rho exactly for D ~ N(0, sigma_D^2)

        # --- Lemma 4.1: Markov bound on P(|Z1-Z2| >= eta) ---
        etas = sigma_D * np.array([0.5, 1.0, 2.0, 3.0])
        for eta in etas:
            lhs = float(2 * norm.sf(eta / sigma_D)) if sigma_D > 0 else 0.0
            rhs = lemma_4_1_probability_bound(rho, p, eta)
            rows.append({"check": "lemma_4_1_markov", "rho": rho, "eta": eta, "lhs": lhs, "rhs": rhs, "holds": lhs <= rhs + 1e-12})

        # --- Lemma 4.2: distribution-function bound, sup over J=[-J_half,J_half] ---
        sigma1 = float(np.sqrt(1.0 + sigma_D ** 2))
        xs = np.linspace(-J_half, J_half, 401)
        sup_diff = float(np.max(np.abs(norm.cdf(xs, scale=sigma1) - norm.cdf(xs))))
        eta_lemma, rhs42 = lemma_4_2_distribution_bound(rho, p, theta, f_sup, eta0)
        rows.append({"check": "lemma_4_2_distribution", "rho": rho, "eta": eta_lemma, "lhs": sup_diff, "rhs": rhs42, "holds": sup_diff <= rhs42 + 1e-9})

        # --- Lemma 6.1 / Theorem 4.3: Hoelder bound for a coherent risk measure ---
        rho_q_star = sigma_D * kappa_p(q_star)
        # Hoelder-extremal dQ/dP ~ |D|^{q-1} sign(D) attains E^Q[D] = ||D||_{q*} exactly
        # when ||dQ/dP||_q = 1; scaling the extremal density to ||dQ/dP||_q = C_rho gives
        # sup_Q E^Q[D] = C_rho * ||D||_{q*}, the equality case of Lemma 6.1's inequality.
        actual_sup_EQ_D = C_rho * rho_q_star
        bound43 = theorem_4_3_coherent_risk_bound(rho_q_star, C_rho)
        rows.append({"check": "theorem_4_3_coherent_risk", "rho": rho, "eta": np.nan, "lhs": actual_sup_EQ_D, "rhs": bound43, "holds": actual_sup_EQ_D <= bound43 + 1e-9})

        # --- Theorem 4.4(i), alpha > 1/2: VaR error via Lemma 6.2 + Theorem 4.3(q=inf) ---
        rho_1 = sigma_D * kappa_p(1.0)  # q* = 1 when q = infinity
        for alpha in config.EXP6_ALPHA_VAR:
            var2, es2 = _gaussian_var_es(0.0, 1.0, alpha)
            var1, es1 = _gaussian_var_es(0.0, sigma1, alpha)
            assert lemma_6_2_var_leq_es(var1, es1) and lemma_6_2_var_leq_es(var2, es2)
            lhs_var = abs(var1 - var2)
            rhs_var = theorem_4_4_var_bound_large_alpha(rho_1, C_rho)
            rows.append({"check": "theorem_4_4_var_large_alpha", "rho": rho, "eta": alpha, "lhs": lhs_var, "rhs": rhs_var, "holds": lhs_var <= rhs_var + 1e-9})

            # --- Theorem 4.4(ii): quantile stability (Lemma 6.3) under a density floor ---
            lam, bound_density = lemma_6_3_quantile_stability(rho, p, c_density, gamma)
            eta_63 = lam * rho ** (p / (p + 1.0))
            nu = 2 * norm.sf(eta_63 / sigma_D) if sigma_D > 0 else 0.0
            condition_ok = bool(2 * nu <= gamma)
            if condition_ok:
                rows.append({"check": "theorem_4_4_var_density", "rho": rho, "eta": alpha, "lhs": lhs_var, "rhs": bound_density, "holds": lhs_var <= bound_density + 1e-9})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp6_risk_extension_checks.csv", index=False)

    _plot(df)

    summary = {}
    for check, sub in df.groupby("check"):
        summary[check] = {"n_checks": int(len(sub)), "fraction_holds": float(sub["holds"].mean())}
    with open(RESULTS_DIR / "exp6_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[exp6] {summary}")
    return summary


def _plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    markov = df[df["check"] == "lemma_4_1_markov"]
    for eta, sub in markov.groupby("eta"):
        sub = sub.sort_values("rho")
        axes[0].loglog(sub["rho"], sub["lhs"], "o-", ms=3, label=f"P(|D|>=eta), eta/sigma fixed")
        break
    coherent = df[df["check"] == "theorem_4_3_coherent_risk"].sort_values("rho")
    axes[1].loglog(coherent["rho"], coherent["lhs"], "o-", label="sup_Q E^Q[D] (attained)")
    axes[1].loglog(coherent["rho"], coherent["rhs"], "--", label="C_rho ||D||_q* bound")
    axes[0].set_title("Lemma 4.1 Markov tail vs rho")
    axes[1].set_title("Theorem 4.3 coherent-risk Hoelder bound")
    for ax in axes:
        ax.set_xlabel("rho")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_risk_extension_checks.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    run()
