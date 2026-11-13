"""Truncation-error lemma (Lemma 5.1, Theorem 3.3) and its sharpness (Proposition
3.8), paper Sec. 3.2 / 5.1, p.10, p.17, p.21.

All quantities here are evaluated in closed form (gamma functions, binomial
sums), not by Monte Carlo: Section 5.1 is a moment computation with explicit
constants, so the per-length bound (5.1) and the tail bound of Lemma 5.1 can
be checked by direct numerical summation over the combinatorics of words.
"""

from __future__ import annotations

from math import floor

import numpy as np
from scipy.special import gammaln, logsumexp


def c_d(d: int) -> float:
    """c_d = 1 + d/2 (Proposition 2.6)."""
    return 1.0 + d / 2.0


def psi_pt(abs_len: int, weighted_len: int, p: float, t: float, d: int) -> float:
    """psi_{p,t}(v) = (p+2)^{|v|} (c_d t)^{||v||/2} / floor(||v||/2)! (Prop. 2.6).

    ``abs_len`` is |v| (word length) and ``weighted_len`` is ||v|| = |v| +
    n_0(v) (time letters counted twice).
    """
    log_val = (
        abs_len * np.log(p + 2)
        + 0.5 * weighted_len * np.log(c_d(d) * t)
        - gammaln(floor(weighted_len / 2) + 1)
    )
    return float(np.exp(log_val))


def word_count_log(m: int, k: int, d: int) -> float:
    """log #{v : |v|=m, n_0(v)=k} = log[C(m,k) d^{m-k}] (Lemma A.2)."""
    return gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + (m - k) * np.log(d)


def per_length_direct_sum(m: int, r: float, t: float, d: int, eps: float, C_star: float = 1.0) -> float:
    """Sigma_{|v|=m} C_* (||v||!)^{-eps} psi_{r,t}(v), summed exactly over n_0(v)=k.

    Uses word_count_log (Lemma A.2) to replace the sum over individual words
    of length m by a sum over k = n_0(v) in {0,...,m}.
    """
    if m == 0:
        return C_star
    logs = []
    for k in range(0, m + 1):
        weighted_len = m + k
        log_psi = (
            m * np.log(r + 2)
            + 0.5 * weighted_len * np.log(c_d(d) * t)
            - gammaln(floor(weighted_len / 2) + 1)
        )
        log_term = word_count_log(m, k, d) - eps * gammaln(weighted_len + 1) + log_psi
        logs.append(log_term)
    return C_star * float(np.exp(logsumexp(logs)))


def per_length_x_t(r: float, t: float, d: int) -> float:
    """x_t = 2(r+2) sqrt(c_d t) (d + 2 sqrt(c_d t)) (per-length bound (5.1))."""
    return 2 * (r + 2) * np.sqrt(c_d(d) * t) * (d + 2 * np.sqrt(c_d(d) * t))


def per_length_claimed_bound(m: int, r: float, t: float, d: int, eps: float, C_star: float = 1.0) -> float:
    """C_* x_t^m / (m!)^eps, the right-hand side of the per-length bound (5.1)."""
    x_t = per_length_x_t(r, t, d)
    if x_t <= 0.0 or m == 0:
        return C_star
    log_val = np.log(C_star) + m * np.log(x_t) - eps * gammaln(m + 1)
    return float(np.exp(log_val))


def tail_bound_direct(M: int, r: float, t: float, d: int, eps: float, C_star: float = 1.0, m_extra: int = 400) -> float:
    """Sum_{m>=M+1} C_* x_t^m/(m!)^eps, the tail of Lemma 5.1's per-length bounds."""
    x_t = per_length_x_t(r, t, d)
    ms = np.arange(M + 1, M + 1 + m_extra)
    logs = np.log(C_star) + ms * np.log(x_t) - eps * gammaln(ms + 1)
    return float(np.exp(logsumexp(logs)))


def e_eps_series(y: float, eps: float, n_terms: int = 400) -> float:
    """E_eps(y) = sum_{j>=0} y^j / (j!)^eps (finite for eps > 0)."""
    js = np.arange(0, n_terms)
    with np.errstate(divide="ignore"):
        logs = js * np.log(y) - eps * gammaln(js + 1)
    logs = logs[np.isfinite(logs)]
    return float(np.exp(logsumexp(logs)))


def tail_bound_closed_form(M: int, B_pq: float, t: float, eps: float, C_star: float = 1.0) -> float:
    """(C_* E_eps(B_pq sqrt t) (B_pq sqrt t)^{M+1} / ((M+1)!)^eps) (Lemma 5.1)."""
    y = B_pq * np.sqrt(t)
    e_eps = e_eps_series(y, eps)
    log_val = np.log(C_star * e_eps) + (M + 1) * np.log(y) - eps * gammaln(M + 2)
    return float(np.exp(log_val))


def sigma_N_sq(t: float, N: int, T: float) -> float:
    """Discretised-variance bound sigma_N(t)^2 = t_k + lambda^2 h (formula (5.3)),

    for t = t_k + lambda h on a uniform grid of step h = T/N; satisfies
    t^2/T <= sigma_N(t)^2 <= t for every N (used to check uniformity in N).
    """
    h = T / N
    k = floor(t / h + 1e-12)
    t_k = k * h
    lam = (t - t_k) / h
    return t_k + lam ** 2 * h


# --- Sharpness example (Proposition 3.8): ell^{(eps)} = sum_n (n!)^{1/2-eps} 1^{(x)n} ---

def sharpness_log_term(j: int, eps: float, sigma2: float) -> float:
    """log of the j-th term of E[X_t(inf,N) - X_t(M,N)] = sum_{2j>M} [(2j)!]^{1/2-eps} (sigma^2/2)^j / j!."""
    return (0.5 - eps) * gammaln(2 * j + 1) + j * np.log(sigma2 / 2.0) - gammaln(j + 1)


def sharpness_tail_expectation_log(M: int, eps: float, sigma2: float, j_extra: int = 2000) -> float:
    """log E[X_t(inf,N) - X_t(M,N)] for the sharpness example, summed over 2j > M."""
    j_start = floor(M / 2) + 1
    js = np.arange(j_start, j_start + j_extra)
    logs = sharpness_log_term_vec(js, eps, sigma2)
    return float(logsumexp(logs))


def sharpness_log_term_vec(js: np.ndarray, eps: float, sigma2: float) -> np.ndarray:
    return (0.5 - eps) * gammaln(2 * js + 1) + js * np.log(sigma2 / 2.0) - gammaln(js + 1)


def sharpness_lower_rhs(t: float, T: float, eps: float, M: int) -> float:
    """RHS of log a_J >= -2 eps J log J - (c_t - 1) J, J = floor(M/2)+1,
    c_t = 2 + |log(t^2/(2T))| (Proposition 3.8 sharpness proof)."""
    J = floor(M / 2) + 1
    c_t = 2.0 + abs(np.log(t ** 2 / (2 * T)))
    if J <= 1:
        return -(c_t - 1) * J
    return -2 * eps * J * np.log(J) - (c_t - 1) * J
