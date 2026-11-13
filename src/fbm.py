"""Shifted Riemann-Liouville fractional Brownian motion (paper Sec. 4.3, p.15).

    X_t = Gamma(H+1/2)^{-1} int_0^t (t + mu - s)^{H-1/2} dW_s,

which admits the signature representation X_t = <ell_t^RL, W-hat_t> with

    ell_t^RL = sum_{n>=0} c_n(t) 0^{(x)n} 1,
    c_n(t)   = (t+mu)^{H-1/2} / Gamma(H+1/2) * (1/2-H)^{n-bar} / (t+mu)^n,

where x^{n-bar} = prod_{k=0}^{n-1} (x+k) is the rising factorial (Pochhammer
symbol). This coefficient family sits on the boundary eps=0 of L^1 (Remark
5.6): the n-th coefficient grows like n! rather than being damped by it, so
the truncation error decays only geometrically, with ratio q_t = t/(t+mu)
(Proposition 4.6), unlike the factorial decay of Sec. 4.2.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.special import gammaln, poch
from scipy.stats import norm

from src import tensor_algebra as ta
from src.tensor_algebra import Series


def rl_coefficient(n: int, t: float, H: float, mu: float) -> float:
    """c_n(t) of the shifted Riemann-Liouville expansion, word 0^{(x)n}1."""
    a = 0.5 - H
    log_gamma_H = gammaln(H + 0.5)
    prefactor = (t + mu) ** (H - 0.5) / np.exp(log_gamma_H)
    return prefactor * poch(a, n) / (t + mu) ** n


def build_ell_RL(t: float, H: float, mu: float, max_len: int) -> Series:
    """ell_t^{RL,<=M} truncated to words 0^{(x)n}1 of length n+1 <= max_len."""
    ell: Series = {}
    for n in range(0, max_len):
        word = (0,) * n + (1,)
        coeff = rl_coefficient(n, t, H, mu)
        if coeff != 0.0:
            ell[word] = coeff
    return ell


def variance_0n1(n: int, t: float) -> float:
    """Var(W-hat_t^{0^{(x)n}1}) = t^{2n+1} / ((n!)^2 (2n+1)) (Sec. 4.3)."""
    log_var = (2 * n + 1) * np.log(t) - 2 * gammaln(n + 1) - np.log(2 * n + 1)
    return float(np.exp(log_var))


def kappa_p(p: float) -> float:
    """kappa_p = E[|Z|^p]^{1/p} for Z ~ N(0,1) (absolute Gaussian moment)."""
    log_abs_moment = (p / 2) * np.log(2) + gammaln((p + 1) / 2) - 0.5 * np.log(np.pi)
    return float(np.exp(log_abs_moment / p))


def proposition_4_6_upper(t: float, M: int, H: float, mu: float, p: float) -> float:
    """Upper bound of Proposition 4.6(i):

        ||X_t(inf,N) - X_t(M,N)||_p
            <= kappa_p (t+mu)^{H+1/2} t^{1/2} / (mu Gamma(H+1/2)) * q_t^M / sqrt(2M+1),

    using |c_n(t)| sqrt(V_n(t)) <= (t+mu)^{H-1/2} t^{1/2} q_t^n / (Gamma(H+1/2) sqrt(2n+1))
    and sum_{n>=M} q_t^n = q_t^M (t+mu)/mu (implementation notes, exp_2).
    """
    q_t = t / (t + mu)
    log_gamma_H = gammaln(H + 0.5)
    const = kappa_p(p) * (t + mu) ** (H + 0.5) * np.sqrt(t) / (mu * np.exp(log_gamma_H))
    return const * q_t ** M / np.sqrt(2 * M + 1)


def proposition_4_6_lower(t: float, M: int, H: float, mu: float) -> float:
    """Lower bound of Proposition 4.6(ii), H != 1/2:

        ||R_t||_2 >= |c_M(t)| sqrt(V_M(t)),

    using the rising-factorial lower bound |a^{M-bar}| >= |a| (M-1)! / (2 sqrt(M)).
    """
    a = 0.5 - H
    if a == 0.0:
        return 0.0
    log_fact = gammaln(M)  # (M-1)!
    log_rising_lower = np.log(abs(a)) + log_fact - np.log(2 * np.sqrt(M))
    log_gamma_H = gammaln(H + 0.5)
    log_prefactor = (H - 0.5) * np.log(t + mu) - log_gamma_H - M * np.log(t + mu)
    log_cM_lower = log_prefactor + log_rising_lower
    cM_lower = np.exp(log_cM_lower)
    return cM_lower * np.sqrt(variance_0n1(M, t))


def geometric_ratio(t: float, mu: float) -> float:
    """q_t = t / (t + mu)."""
    return t / (t + mu)


def reference_process(
    dW: np.ndarray, T: float, H: float, mu: float, grid_index: int
) -> np.ndarray:
    """X_t(infinity, N): the untruncated signature functional evaluated on the
    piecewise-linear interpolation of W, i.e. the full (non-truncated) sum of
    c_n(t) * W-hat_t^{0^{(x)n}1}(N) over all n.

    Using (t+mu-s)^{H-1/2} = (t+mu)^{H-1/2} sum_n [a^{n-bar}/n!] (s/(t+mu))^n
    with a=1/2-H, and W-hat_t^{0^{(x)n}1}(N) = int_0^t (s^n/n!) dW-tilde_s
    (signature of the time-extended path against word 0^n1), the infinite sum
    reconstructs exactly

        X_t(infinity,N) = Gamma(H+1/2)^{-1} int_0^t (t+mu-s)^{H-1/2} dW-tilde_s,

    computed in closed form segment by segment since dW-tilde is piecewise
    constant on the uniform grid (no truncation level M, no dense tensors).

    Parameters
    ----------
    dW : array (n_paths, N) grid increments of W on [0, T]
    grid_index : the grid point index K such that t = K * (T / N)
    """
    n_paths, N = dW.shape
    h = T / N
    K = grid_index
    if K == 0:
        return np.zeros(n_paths)
    t = K * h
    k = np.arange(1, K + 1)
    t_km1 = (k - 1) * h
    t_k = k * h
    # exact segment integral of (t+mu-s)^{H-1/2} ds over [t_{k-1}, t_k]
    exponent = H + 0.5
    weights = ((t + mu - t_km1) ** exponent - (t + mu - t_k) ** exponent) / exponent
    log_gamma_H = gammaln(H + 0.5)
    return (dW[:, :K] / h) @ weights / np.exp(log_gamma_H)
