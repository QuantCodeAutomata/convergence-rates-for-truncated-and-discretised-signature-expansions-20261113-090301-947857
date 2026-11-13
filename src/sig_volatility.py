"""Signature volatility model and the log-price map Phi (paper Sec. 4.4 / 7).

    dS_t = S_t sigma_t dW^S_t,  sigma_t = <ell, W-hat_t>,  W^S = rho W + sqrt(1-rho^2) W^perp,

with ell supported on words over the alphabet {0 (time), 1 (W)} only (zero on
words containing the letter 2 = W^perp, so that ell is viewed in L^1(R^3)).
The log-price X_t = log S_t is then the linear functional

    X_t = <Phi(ell), W-hat_t^*>,
    Phi(ell) := log(S_0) empty - (alpha*/2) ell^{shuffle 2} 0 + sum_i alpha_i pi_i(ell),
    pi_i(ell) := ell.i - (1/2) 1_{i != 0} (ell|_i).0,

of the signature of (t, W_t, W_t^perp) (Lemma 7.1, Sec. 7). Truncation commutes
with Phi (Lemma 7.2): Phi(ell)^{<=M} = Phi(ell^{<=M})^{<=M}, so the pricing
scheme of Proposition 7.3 only ever needs ell^{<=M}.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from src import tensor_algebra as ta
from src.tensor_algebra import Series


def build_ell_vol(ell_empty: float, ell0: float, ell1: float) -> Series:
    """Level-<=1 volatility functional ell = ell_empty + ell0 (0) + ell1 (1).

    Restricting to level <=1 words makes sigma_t = ell_empty + ell0 t + ell1 W_t
    an explicit function of (t, W_t), which keeps the reference simulation of
    S_t exact up to the Euler discretisation (no extra signature machinery is
    needed to generate the reference path).
    """
    ell: Series = {}
    if ell_empty != 0.0:
        ell[()] = ell_empty
    if ell0 != 0.0:
        ell[(0,)] = ell0
    if ell1 != 0.0:
        ell[(1,)] = ell1
    return ell


def pi_i(ell: Series, letter: int, max_len: int) -> Series:
    """pi_i(ell) = ell.i - (1/2) 1_{i!=0} (ell|_i).0 (Lemma 7.1)."""
    term = ta.append_letter(ell, letter, max_len)
    if letter != 0:
        restricted = ta.restrict_last(ell, letter)
        correction = ta.series_scale(ta.append_letter(restricted, 0, max_len), -0.5)
        term = ta.series_add(term, correction)
    return term


def phi_map(ell: Series, rho: float, log_s0: float, max_len: int) -> Series:
    """Phi(ell) for the signature volatility model, alphabet {0,1,2}.

    alpha = (0, rho, sqrt(1-rho^2)) indexes (time, W, W^perp); alpha* :=
    sum_{i>=1} alpha_i^2 = rho^2 + (1-rho^2) = 1 identically, consistent with
    W^S having unit quadratic variation regardless of rho.
    """
    alpha = (0.0, rho, float(np.sqrt(max(1.0 - rho ** 2, 0.0))))
    alpha_star = alpha[1] ** 2 + alpha[2] ** 2

    shuffle_sq = ta.shuffle_product(ell, ell, max_len)
    out = ta.empty_series(log_s0)
    out = ta.series_add(out, ta.series_scale(ta.append_letter(shuffle_sq, 0, max_len), -alpha_star / 2))
    for i, a_i in enumerate(alpha):
        if a_i == 0.0:
            continue
        out = ta.series_add(out, ta.series_scale(pi_i(ell, i, max_len), a_i))
    return out


def simulate_log_price_reference(
    dW: np.ndarray,
    dW_perp: np.ndarray,
    ell_empty: float,
    ell0: float,
    ell1: float,
    rho: float,
    log_s0: float,
    T: float,
) -> np.ndarray:
    """Fine-grid Euler-Maruyama reference for X_t = log S_t.

    sigma_t = ell_empty + ell0 t + ell1 W_t is Markovian in (t, W_t) because
    ell is restricted to level <=1, so the Euler scheme for
    dX_t = -sigma_t^2/2 dt + sigma_t dW^S_t is exact up to the usual
    discretisation bias, independently of the signature machinery used for
    the truncated scheme.
    """
    n_paths, N_ref, _ = dW.shape
    h = T / N_ref
    W = np.zeros(n_paths)
    X = np.full(n_paths, log_s0)
    X_path = np.empty((n_paths, N_ref + 1))
    X_path[:, 0] = X
    sqrt_term = np.sqrt(max(1.0 - rho ** 2, 0.0))
    for k in range(N_ref):
        t_k = k * h
        sigma_k = ell_empty + ell0 * t_k + ell1 * W
        dWS_k = rho * dW[:, k, 0] + sqrt_term * dW_perp[:, k, 0]
        X = X - 0.5 * sigma_k ** 2 * h + sigma_k * dWS_k
        W = W + dW[:, k, 0]
        X_path[:, k + 1] = X
    return X_path


def put_price_mc(log_price: np.ndarray, K: float) -> Tuple[float, float]:
    """Monte Carlo put price E[(K - exp(X))^+] and its standard error."""
    payoff = np.maximum(K - np.exp(log_price), 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(payoff.size))
