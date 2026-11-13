"""Extensions of the strong-error bound to probability, distribution
functions and risk measures (Sec. 4.1, p.11-12; proofs in Sec. 6, p.26-27).

Every function here takes rho = ||Z_1 - Z_2||_p (a stand-in for the strong
total error C * E_a(M,N) of Corollary 3.6, applied to Z_1 = g(X_t) and
Z_2 = g(X_t(M,N))) as input, and returns the corresponding bound from the
cited lemma/theorem. This isolates the chain of real-analysis inequalities
(Markov, Hoelder, quantile stability) from the signature-expansion error
itself, which is verified independently in the other experiments.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def lemma_4_1_probability_bound(rho: float, p: float, eta: float) -> float:
    """Lemma 4.1: P(|Z_1-Z_2| >= eta) <= eta^{-p} rho^p, via Markov's inequality."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    return (rho / eta) ** p


def lemma_4_2_distribution_bound(rho: float, p: float, theta: float, f_sup: float, eta0: float) -> Tuple[float, float]:
    """Lemma 4.2: sup_{x in J} |F_1(x)-F_2(x)| <= (2||f^g||_I + 1 + eta0^{-1}) rho^theta,
    valid when eta = rho^{p/(p+1)} satisfies p/(p+1) >= theta (Lemma B.3 + density bound).

    Returns (eta, bound).
    """
    if not 0.0 < theta <= p / (p + 1) + 1e-9:
        raise ValueError("theta must satisfy 0 < theta <= p/(p+1)")
    eta = rho ** (p / (p + 1))
    bound = (2.0 * f_sup + 1.0 + 1.0 / eta0) * rho ** theta
    return eta, bound


def lemma_b3_distribution_split(F1_plus: float, F1_minus: float, prob_tail: float) -> float:
    """Lemma B.3: |F_1(x)-F_2(x)| <= [F_1(x+eta)-F_1(x-eta)] + P(|Z_1-Z_2|>eta)."""
    return max(F1_plus - F1_minus, 0.0) + prob_tail


def lemma_6_1_holder_coherent(rho_q_star: float, C_rho: float) -> float:
    """Lemma 6.1: |rho_risk(Z)| <= C_rho ||Z||_{q*}, q* = q/(q-1) (Hoelder)."""
    return C_rho * rho_q_star


def theorem_4_3_coherent_risk_bound(rho_q_star: float, C_rho: float) -> float:
    """Theorem 4.3: |rho_risk(Z_1)-rho_risk(Z_2)| <= rho_risk(|Z_1-Z_2|) <= C_rho ||Z_1-Z_2||_{q*}."""
    return lemma_6_1_holder_coherent(rho_q_star, C_rho)


def lemma_6_2_var_leq_es(var_alpha: float, es_alpha: float) -> bool:
    """Lemma 6.2: VaR_alpha[Z] <= ES_alpha[Z] for alpha in (0,1)."""
    return var_alpha <= es_alpha + 1e-9


def theorem_4_4_var_bound_large_alpha(rho: float, C_rho: float) -> float:
    """Theorem 4.4(i), alpha > 1/2: |VaR_alpha[Z_1]-VaR_alpha[Z_2]|
    <= |ES_alpha[Z_1]-ES_alpha[Z_2]| <= C_rho rho (Lemma 6.2 + Theorem 4.3, q=infty)."""
    return C_rho * rho


def lemma_6_3_quantile_stability(rho: float, p: float, c: float, gamma: float) -> Tuple[float, float]:
    """Lemma 6.3: with eta = lambda rho^{p/(p+1)} and nu = P(|Z_1-Z_2|>=eta)
    <= lambda^{-p} rho^{p/(p+1)}, choose lambda so that 2 nu <= gamma (density
    lower bound c on B); then
        |VaR_alpha[Z_1]-VaR_alpha[Z_2]| <= (lambda + 2/(c lambda^p)) rho^{p/(p+1)}.

    Returns (lambda, bound) at the lambda solving 2 nu = gamma exactly.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    exponent = p / (p + 1)
    lam = (2.0 * rho ** exponent / gamma) ** (1.0 / p)
    bound = (lam + 2.0 / (c * lam ** p)) * rho ** exponent
    return lam, bound


def theorem_4_4_var_bound_density(rho: float, p: float, c: float, gamma: float) -> float:
    """Theorem 4.4(ii): VaR error bound under a local density lower bound c on B."""
    _, bound = lemma_6_3_quantile_stability(rho, p, c, gamma)
    return bound


def e_a_rate(rho_M: float, rho_N: float) -> float:
    """E_a(M,N) = r_M^a + N^{-a/2}, the total-error rate of Corollary 3.6 /
    Theorem 3.9 that every extension bound above is ultimately applied to."""
    return rho_M + rho_N
